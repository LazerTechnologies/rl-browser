import re
import time
from bs4 import BeautifulSoup, Comment, Tag
from bs4.element import NavigableString
from playwright.async_api import ElementHandle, Locator, Page, expect


def is_visible_bs(element):
    """
    Checks if an element is likely visible based on common HTML attributes and inline styles.
    This is not a perfect solution as it does not parse external CSS files.
    """
    if element is None:
        return False

    # Check for inline 'display: none' or 'visibility: hidden' styles
    style = element.get("style", "")
    if "display" in style and "none" in style:
        return False
    if "visibility" in style and "hidden" in style:
        return False

    # Check for a 'hidden' attribute
    if element.get("hidden") is not None:
        return False

    # Recursively check parent elements
    parent = element.parent
    if parent is not None and parent.name != "[document]":
        return is_visible_bs(parent)

    return True


class DOMSimplifier:
    def __init__(self):
        self.interactive_tags = {
            "a",
            "button",
            "input",
            "select",
            "textarea",
            "details",
            "summary",
            "iframe",
        }
        # Tags that define structure/grouping (keep these even if not interactive)
        self.semantic_tags = {
            "body",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "form",
            "label",
            "table",
            "thead",
            "tbody",
            "tr",
            "td",
            "th",
            "main",
            "header",
            "footer",
            "nav",
            "section",
            "article",
        }
        self.ignore_tags = {
            "script",
            "style",
            "meta",
            "noscript",
            "link",
            "svg",
            "path",
            "br",
        }

        self.element_map = {}
        self.counter = 1

    def clean_dom(self, soup):
        for tag in soup(self.ignore_tags):
            tag.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        for tag in soup.find_all("input", type="hidden"):
            tag.decompose()
        return soup

    def get_own_text(self, soup_element: Tag):
        """Gets text actually inside this tag, NOT inside its children."""
        text = ""
        for child in soup_element.children:
            if isinstance(child, NavigableString) and not isinstance(child, Comment):
                trimmed = child.strip()
                if trimmed:
                    text += trimmed + " "
        text = text.strip()
        if not text:
            aria_label = soup_element.get("aria-label")
            aria_str = str(aria_label) if aria_label else ""
            text = (aria_str + " " + ",".join(soup_element.get("class") or [])).strip()

        return text

    def get_xpath(self, elem):
        """Generates a unique XPath for a BeautifulSoup element."""
        path = []
        while elem:
            # Use ID if available for shorter XPath
            if elem.attrs.get("id"):
                path.append(f"//*[@id='{elem['id']}']")
                break

            # Determine the position among siblings of the same tag type
            siblings = (
                elem.find_parent().find_all(elem.name, recursive=False)
                if elem.parent
                else []
            )
            if len(siblings) > 1:
                # Find the index of the current element (1-based index for XPath)
                index = siblings.index(elem) + 1
                path.append(f"{elem.name}[{index}]")
            else:
                path.append(elem.name)

            elem = elem.parent
            if not elem or elem.name == "[document]":
                break

        # Reverse the list and join with /
        return "/".join(path[::-1])

    async def process_node(self, locator: Locator, soup_element: Tag, depth=0):
        if isinstance(soup_element, NavigableString) or not is_visible_bs(soup_element):
            return ""

        tag_name = soup_element.name
        own_text = self.get_own_text(soup_element)

        is_interactive = (tag_name in self.interactive_tags) or (
            soup_element.get("aria-label") is not None
        )
        is_semantic = tag_name in self.semantic_tags
        has_direct_text = len(own_text) > 0

        should_print = (
            is_interactive
            or is_semantic
            or has_direct_text
            or (soup_element.get("aria-label") is not None)
        )

        # print(f"> Processing <{tag_name}> <{own_text}>  should_print={should_print}")

        element_handle = None
        if should_print and is_interactive:
            handles = await locator.element_handles()
            element_handle = handles[-1] if handles else None
            if element_handle:
                box = await element_handle.bounding_box()
                if box:
                    in_viewport = (
                        box["y"] < self.viewport["height"]
                        and box["y"] + box["height"] > 0
                        and box["x"] < self.viewport["width"]
                        and box["x"] + box["width"] > 0
                    )
                    if not in_viewport:
                        return ""

        # Important Attributes
        attrs = []
        # ID generation - ONLY assign ID if we're going to print this node AND it's interactive
        node_id_str = ""

        if should_print and is_interactive and element_handle:
            node_id = self.counter
            self.element_map[node_id] = element_handle
            node_id_str = f"[{node_id}] "
            self.counter += 1

        # Collect useful attributes using BeautifulSoup
        if soup_element.get("aria-label"):
            attrs.append(f'aria="{soup_element.get("aria-label")}"')
        if soup_element.get("placeholder"):
            attrs.append(f'plh="{soup_element.get("placeholder")}"')
        if soup_element.get("alt"):
            attrs.append(f'alt="{soup_element.get("alt")}"')
        if soup_element.get("name"):
            attrs.append(f'name="{soup_element.get("name")}"')
        if tag_name == "input":
            attrs.append(f'type="{soup_element.get("type") or "text"}"')

        output = ""
        if should_print:
            indent = "  " * depth
            attr_str = " " + " ".join(attrs) if attrs else ""
            text_str = f' "{own_text}"' if own_text else ""
            output += f"{indent}{node_id_str}{tag_name}{attr_str}{text_str}\n"
            next_depth = depth + 1
        else:
            # PASS-THROUGH: This is a generic div/span.
            # Don't print it, but process its children at the SAME depth.
            next_depth = depth

        # 3. Recurse children using BeautifulSoup for iteration
        # O(1) sibling indexing
        from collections import defaultdict

        seen = defaultdict(int)
        for child in soup_element.children:
            if not hasattr(child, "name"):
                continue
            seen[child.name] += 1
            idx = seen[child.name]
            if idx > 1:
                child_locator = locator.locator(f"> {child.name}:nth-of-type({idx})")
            else:
                child_locator = locator.locator(f"> {child.name}")
            output += await self.process_node(child_locator, child, next_depth)

        return output

    async def parse(self, page: Page) -> tuple[str, dict[str, ElementHandle]]:
        self.counter = 1
        self.element_map = {}

        self.viewport = page.viewport_size
        assert self.viewport is not None

        # Get the DOM content and parse with BeautifulSoup
        dom_content = await page.content()
        soup = BeautifulSoup(dom_content, "html.parser")
        soup = self.clean_dom(soup)

        # Get body from both Playwright and BeautifulSoup
        body_locator = page.locator("body")
        body_soup = soup.body if soup.body else soup

        assert body_locator is not None
        assert body_soup is not None

        print("> Starting DOM parsing...")
        result = await self.process_node(body_locator, body_soup)

        return (result, self.element_map)
