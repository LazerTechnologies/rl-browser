import json
import ast
import random
import asyncio
import base64
import os
import shutil
import time
import ua_generator
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from playwright.async_api import Page, async_playwright, ElementHandle
from pydantic import BaseModel

from utils import DOMSimplifier

# from .index import RAGSystem
from agent import fetch_query_for_rag, get_reply, summarize_text


@dataclass
class Action:
    action_type: str
    params: Optional[Dict[str, Union[str, int]]]


class PlaywrightExecutor:
    def __init__(self, page: Page):
        self.page = page
        self.elements = {}

    async def get_dom(self) -> str:
        return await self.page.content()

    async def execute_action(
        self, action_str: str, elements: Dict[str, ElementHandle]
    ) -> None | str:
        """Execute a Playwright action from a string command."""
        action = self.parse_action(action_str)
        self.elements = elements
        # start_time = time.time()
        element = action.params["uid"] if "uid" in action.params else None
        if action.action_type == "click":
            if not element:
                raise ValueError(f"Element with id {action.params['uid']} not found")
            try:
                await self._execute_click(element)
            except Exception as e:
                if "waiting for element to be visible" in str(e):
                    return f"Element not interactable"
                else:
                    raise e

        elif action.action_type == "text_input":
            if not element:
                raise ValueError(f"Element with id {action.params['uid']} not found")
            await self._execute_change(element, action.params["text"])
        elif action.action_type == "change":
            if not element:
                raise ValueError(f"Element with id {action.params['uid']} not found")
            await self._execute_change(element, action.params["value"])
        elif action.action_type == "scroll":
            await self._execute_scroll(int(action.params["x"]), int(action.params["y"]))
        elif action.action_type == "submit":
            if not element:
                raise ValueError(f"Element with id {action.params['uid']} not found")
            await self._execute_submit(element)
        elif action.action_type == "back":
            await self.page.go_back()
        elif action.action_type == "enter":
            await self.page.keyboard.press("Enter")
        elif action.action_type == "load":
            await self._execute_load(action.params["url"])
        elif action.action_type == "nothing":
            pass
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

        # print("> Action time taken:", time.time() - start_time)

    def parse_action(self, action_str: str) -> Action:
        """Parse an action string into an Action object."""
        if action_str == "back":
            return Action(action_type="back", params={})
        if action_str == "enter":
            return Action(action_type="enter", params={})
        if action_str == "nothing":
            return Action(action_type="nothing", params={})
        action_type = action_str[: action_str.index("(")]
        params_str = action_str[action_str.index("(") + 1 : action_str.rindex(")")]
        params = {}
        if params_str:
            param_pairs = params_str.split(",")
            for pair in param_pairs:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                params[key] = value
        return Action(action_type=action_type, params=params)

    async def find_element_by_reference(self, ref_id: int) -> ElementHandle:
        selector = self.elements.get(int(ref_id))

        if not selector:
            raise ValueError(
                f"Element with reference id {ref_id} not found", self.elements
            )

        element_handle = self.elements.get(int(ref_id), None)
        if not element_handle:
            raise ValueError(f"Could not find element matching: {selector}")

        return element_handle

    async def _execute_click(self, element_id: int) -> None:
        """Execute a click action."""
        element_handle = await self.find_element_by_reference(element_id)
        await element_handle.click(timeout=5000)

    async def _execute_text_input(self, element_id: int, text: str) -> None:
        """Execute a text input action."""
        element_handle = await self.find_element_by_reference(element_id)
        await element_handle.click(timeout=5000)
        await self.page.keyboard.type(text, delay=100)

    async def _execute_change(self, element_id: int, value: str) -> None:
        """Execute a change action."""
        # TODO: May need to "click" the x+y of the element to set focus first
        element_handle = await self.find_element_by_reference(element_id)
        await element_handle.focus()
        await self.page.keyboard.down("Meta")
        await self.page.keyboard.press("A")
        await self.page.keyboard.up("Meta")
        await self.page.keyboard.type(value, delay=100)

    async def _execute_load(self, url: str) -> None:
        """Execute a load action."""
        await self.page.goto(url)

    async def _execute_scroll(self, x: int, y: int) -> None:
        """Execute a scroll action."""
        await self.page.evaluate(f"window.scrollTo({x}, {y})")
        await self.page.wait_for_timeout(1000)

    # async def _execute_submit(self, element_id: int) -> None:
    #     """Execute a submit action."""
    #     x, y = await self.find_element_by_reference(element_id)
    #     await self.page.mouse.click(x, y, delay=100)


class WebScraper:
    def __init__(self, task, start_url, output_model: BaseModel, callback=None):
        self.logs = []
        self.log_callback = callback

        self._log("Initializing WebScraper...")
        self.task = task
        self.start_url = start_url
        index_path = "output/index"
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        # self.rag = RAGSystem(index_path="output/index")
        self.output_model = output_model
        self.browser = None
        self.iteration_count = 0
        self._log("Done initializing WebScraper")

    def _log(self, message):
        self.logs.append(message)
        if self.log_callback:
            self.log_callback(message)

    async def main(self, p):
        # locally
        self._log("Starting browser...")
        self.browser = await p.chromium.launch(
            headless=True,
        )

        user_agent = ua_generator.generate(
            device="desktop",
        )

        size = {"width": 1920, "height": 1080}

        context = await self.browser.new_context(
            record_video_dir="videos/",
            record_video_size=size,
        )
        page = await context.new_page()
        await page.set_viewport_size(size)
        next_task = (
            "Find the website to visit."
            if "google.com" in self.start_url
            else "Figure out what to do on the website."
        )
        next_action = f'load(url="{self.start_url}")'
        second_action = None
        max_iterations = 30
        self.iteration_count = 0
        state = [
            {
                "role": "user",
                "content": f"""Overall goal: {self.task}. Try to find the following information in your search: {self.output_model.model_json_schema()['$defs'] if '$defs' in self.output_model.model_json_schema() else self.output_model.model_json_schema()['properties']}""",
            },
            {
                "role": "assistant",
                "content": "Okay. Let's get started.",
            },
        ]
        simplifier = DOMSimplifier()
        executor = PlaywrightExecutor(page)
        elements, elements_id_map = await simplifier.parse(executor.page)
        while next_task and self.iteration_count < max_iterations:
            self._log(f"> Executing action {next_action}")
            error_response = await executor.execute_action(next_action, elements_id_map)
            time.sleep(1)
            # TODO: temp disable second action
            # if second_action and not error_response:
            #     self._log(f"> Executing second action {second_action}")

            #     # elements, elements_id_map = await simplifier.parse(executor.page)
            #     error_response = await executor.execute_action(
            #         second_action, elements_id_map
            #     )
            #     time.sleep(1)

            # save
            elements, elements_id_map = await simplifier.parse(executor.page)
            print("> Parsed DOM, preparing to send to AI...")
            await executor.page.screenshot(path="screenshot.png", scale="css")
            # self.rag.add_document(
            #     elements,
            #     {"url": page.url, "timestamp": datetime.now().isoformat()},
            # )
            # self._log(f"> Elements on page: {elements}")

            # with open("screenshot.png", "rb") as img_file:
            #     img = base64.b64encode(img_file.read()).decode("utf-8")

            if error_response:
                self._log(f"> Error response: {error_response}")
                state.append(
                    {
                        "role": "user",
                        "content": f"Error encountered during action execution: {error_response}",
                    }
                )
            else:
                state.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Current URL: {page.url}"},
                            {
                                "type": "text",
                                "text": "Elements on screen:\n" + elements,
                            },
                            # {
                            #     "type": "image_url",
                            #     "image_url": {
                            #         # "type": "base64",
                            #         # "media_type": "image/png",
                            #         "url": f"data:image/png;base64,{img}",
                            #     },
                            # },
                        ],
                        # ],
                    }
                )
            self._log("> Getting reply from AI...")
            start_time = datetime.now()
            reply = get_reply(state)
            self._log(
                f"> AI time taken: {(datetime.now() - start_time).total_seconds()}"
            )

            next_task, next_action, second_action = (
                reply["next_task"],
                reply["next_action"],
                reply.get("next_action_2"),
            )
            self._log(
                f"> Next_task: {next_task}, Next action: {next_action}, Second action: {second_action}"
            )
            state.append(
                {
                    "role": "assistant",
                    "content": f"Next task: {next_task}. Next action: {next_action}",
                }
            )

            if next_action == "nothing" or next_action is None:
                self._log("> No further action required.")
                self.iteration_count += 1000
            else:
                self.iteration_count += 1
        return page, context

    async def run(self):
        async with async_playwright() as p:
            start = time.time()
            page, context = await self.main(p)

            # rag_query = fetch_query_for_rag(self.task)
            # self._log(f"> Querying RAG for task: {rag_query}")
            # docs = [a["text"] for a in self.rag.query(rag_query)]
            # answer = summarize_text(self.task, docs, self.output_model)
            # self._log(f"> Answer: {answer}")
            self._log(f"> Total time taken: {time.time() - start}")

            try:
                await context.close()
            except Exception as e:
                raise Warning(e)

        # return answer
