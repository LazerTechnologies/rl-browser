import re
import json
import ast
from pydantic import BaseModel
import litellm
from litellm import completion
import os
from functools import lru_cache
from datetime import datetime

# Configure LiteLLM with OpenAI-compatible API
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

if LITELLM_BASE_URL:
    litellm.api_base = LITELLM_BASE_URL
if LITELLM_API_KEY:
    litellm.api_key = LITELLM_API_KEY

base_path = os.path.dirname(os.path.abspath(__file__))
RAW_SYSTEM_PROMPT = open(os.path.join(base_path, "agent_prompt.txt")).read()

# litellm.set_verbose = True
litellm.modify_params = True


def parse_text(text):
    next_action_pattern = r"<next_action-1>\n(.*?)\n</next_action-1>"
    next_action2_pattern = r"<next_action-2>\n(.*?)\n</next_action-2>"
    explanation_pattern = r"<explanation>\n(.*?)\n</explanation>"
    next_task_pattern = r"<next_task>\n(.*?)\n</next_task>"

    next_action_match = re.search(next_action_pattern, text, re.DOTALL)
    next_action2_match = re.search(next_action2_pattern, text, re.DOTALL)
    explanation_match = re.search(explanation_pattern, text, re.DOTALL)
    next_task_match = re.search(next_task_pattern, text, re.DOTALL)

    result = {
        "next_action": next_action_match.group(1) if next_action_match else None,
        "next_action_2": (next_action2_match.group(1) if next_action2_match else None),
        "explanation": explanation_match.group(1) if explanation_match else None,
        "next_task": next_task_match.group(1) if next_task_match else None,
    }

    return result


def is_valid_json(string: str) -> bool:
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False


def clean_up_json(string: str) -> str:
    def extract_json_from_string(string):
        start_index = string.find("{")
        end_index = string.rfind("}")
        if start_index != -1 and end_index != -1:
            return string[start_index : end_index + 1]
        return ""

    cleaned = (
        extract_json_from_string(string)
        .strip()
        .replace("\n", "")
        .replace('\\"', '"')
        .replace("```", "")
        .replace("json", "")
    )

    # Check if there's a missing "}" at the end and add it
    if cleaned.count("{") > cleaned.count("}"):
        cleaned += "}"

    if not is_valid_json(cleaned):
        try:
            cleaned = json.dumps(ast.literal_eval(cleaned))
        except (ValueError, SyntaxError):
            raise ValueError("String not valid", cleaned)
    return cleaned


def get_reply(state, model: str = "gemini/gemini-2.5-pro") -> str:
    """
    Get a reply from the LLM using OpenAI-compatible API.

    Args:
        state: The conversation state/history
        model: The model name (will be prefixed with "openai/")
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    SYSTEM_PROMPT = f"{RAW_SYSTEM_PROMPT}\n\nToday's date: {today_date}"

    # Prefix model with "openai/" for LiteLLM
    full_model_name = f"openai/{model}"

    reply = (
        completion(
            model=full_model_name,
            # max_tokens=int(256 * 1.5),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + state,
            temperature=0.5,
        )
        .choices[0]
        .message.content
    )

    return parse_text(reply)


def summarize_text(
    prompt: str, documents: list, schema: str, model: str = "gpt-4"
) -> str:
    """
    Summarize documents using OpenAI-compatible API.

    Args:
        prompt: The prompt for summarization
        documents: List of documents to summarize
        schema: The JSON schema to use for output
        model: The model name (will be prefixed with "openai/")
    """
    full_model_name = f"openai/{model}"

    return json.loads(
        clean_up_json(
            (
                "{"
                + completion(
                    model=full_model_name,
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""Summarize the following documents for this prompt in JSON format.
                
                Prompt: {prompt}
                
                Return using this schema: {schema}""",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text,
                                }
                                for text in documents
                            ],
                        },
                        {"role": "assistant", "content": "{"},
                    ],
                )
                .choices[0]
                .message.content
            )
        )
    )


def fetch_query_for_rag(task: str, model: str = "gemini/gemini-2.5-flash") -> str:
    """
    Generate a RAG query using OpenAI-compatible API.

    Args:
        task: The task description
        model: The model name (will be prefixed with "openai/")
    """
    full_model_name = f"openai/{model}"

    response = clean_up_json(
        "{"
        + completion(
            model=full_model_name,
            max_tokens=256,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": "Generate a simple keyword/phrase query for a RAG system based on the following task. Return the query as JSON with 'query' key. The query should help fetch documents relevant to the task: "
                    + task,
                },
                {"role": "assistant", "content": "{"},
            ],
        )
        .choices[0]
        .message.content
    )
    return json.loads(response)["query"]


@lru_cache(maxsize=128, typed=True)
def find_schema_for_query(query: str, model: str = "gemini/gemini-2.5-flash") -> str:
    """
    Find a JSON schema for a given query using OpenAI-compatible API.

    Args:
        query: The query to generate schema for
        model: The model name (will be prefixed with "openai/")
    """
    full_model_name = f"openai/{model}"

    return clean_up_json(
        completion(
            model=full_model_name,
            temperature=0.5,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": """You're an expert in data science. You're helping a colleague form JSON schemas for their data. You're given a query and asked to find the schema for it.
                 
                 Example:
                 Query: Find 2 recent issues from PyTorch repository.
                 Schema: {'properties': {'date': {'title': 'Date', 'type': 'string'}, 'title': {'title': 'Title', 'type': 'string'}, 'author': {'title': 'Author', 'type': 'string'}, 'description': {'title': 'Description', 'type': 'string'}}, 'required': ['date', 'title', 'author', 'description'], 'title': 'IssueModel', 'type': 'object'}
                 
                 Example:
                 Query: Find 5 events happening in Bangalore this week.
                 Schema: {'properties': {'name': {'title': 'Name', 'type': 'string'}, 'date': {'title': 'Date', 'type': 'string'}, 'location': {'title': 'Location', 'type': 'string'}}, 'required': ['name', 'date', 'location'], 'title': 'EventsModel', 'type': 'object'}""",
                },
                {
                    "role": "user",
                    "content": f"""Find the schema for the following query: {query}.""",
                },
            ],
        )
        .choices[0]
        .message.content
    )
