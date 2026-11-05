"""Parse tool calls out of model output and out of dataset gold text."""

import ast
import json
import re
from dataclasses import dataclass

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


@dataclass
class ParsedCalls:
    calls: list[dict]  # well formed calls: {"name": str, "arguments": dict}
    n_blocks: int  # tool_call blocks found, whether or not they parsed


def _well_formed(obj) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and isinstance(obj.get("arguments"), dict)
    )


def _strip_literal_newlines(body: str) -> str:
    # Some gold blocks wrap the call in the two characters backslash and n.
    body = body.strip()
    while body.startswith("\\n"):
        body = body[2:].lstrip()
    while body.endswith("\\n"):
        body = body[:-2].rstrip()
    return body


def _load(body: str, lenient: bool):
    if lenient:
        body = _strip_literal_newlines(body)
    try:
        return json.loads(body)
    except ValueError:
        pass
    if lenient:
        # Part of the dataset writes calls as Python dicts with single quotes.
        try:
            return ast.literal_eval(body)
        except (ValueError, SyntaxError):
            pass
    return None


def parse_calls(text: str, lenient: bool = False) -> ParsedCalls:
    """Extract calls from <tool_call> blocks.

    Strict mode accepts only JSON, which is what a model must emit.
    Lenient mode also accepts the Python style the dataset uses for some gold labels.
    """
    blocks = _BLOCK.findall(text)
    calls = [obj for obj in (_load(b, lenient) for b in blocks) if _well_formed(obj)]
    return ParsedCalls(calls=calls, n_blocks=len(blocks))
