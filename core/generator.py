"""LLM-assisted test-case drafting: turns a plain-language description of a
test scenario into a structured TestCase (trigger prompt, expected tool
sequence, golden response, pass threshold) for the user to review and refine.
"""
from __future__ import annotations

import inspect
import json
import os
import re

from google import genai

from agent.tools import ALL_TOOLS, ORDERS, INVENTORY
from core.retry import with_retry_sync
from models.schemas import ExpectedToolCall, TestCase

GENERATOR_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash-lite")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _client


def _tool_catalog() -> str:
    lines = []
    for tool in ALL_TOOLS:
        doc = (inspect.getdoc(tool) or "").splitlines()[0]
        lines.append(f"- {tool.__name__}: {doc}")
    return "\n".join(lines)


_GENERATOR_PROMPT = """You are helping a QA engineer draft a test case for an AI customer-support agent.

The agent under test can call these tools (in any order it deems necessary):
{tool_catalog}

Sample data it has access to:
- Orders: {order_ids} (e.g. ORD-1002 was placed more than 30 days ago and is OUTSIDE the refund window; ORD-1001 and ORD-1003 are within it)
- Inventory SKUs: {sku_ids} (e.g. SKU-310 is currently OUT OF STOCK; the others are in stock)

The QA engineer described the scenario they want to test like this:
"{description}"

Draft a complete test case for this scenario. Respond with ONLY a JSON object in
exactly this shape, no markdown fences, no extra commentary:

{{
  "name": "<short descriptive title, under 8 words>",
  "trigger_prompt": "<a realistic first-person message a customer would send that should trigger this scenario>",
  "expected_tools": ["<tool_name_1>", "<tool_name_2>", ...],
  "golden_response": "<1-3 sentences describing what an ideal final answer from the agent should communicate, written so it can be compared against the agent's actual reply>",
  "pass_threshold": <integer 0-100, the minimum semantic-match score to count as a pass; default to 70 unless the scenario calls for stricter or looser grading>
}}

Notes:
- "expected_tools" must be an ORDERED list using ONLY the exact tool names from the catalog above, reflecting the sequence a correctly-behaving agent should follow for this scenario (e.g. it must look up an order before refunding it, and must check refund eligibility before processing a refund).
- Pick concrete order IDs / SKUs from the sample data above so the scenario is testable and deterministic.
"""


def _parse_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def generate_test_case(description: str) -> TestCase:
    """Draft a TestCase from a plain-language scenario description using Gemini."""
    valid_tool_names = {t.__name__ for t in ALL_TOOLS}

    prompt = _GENERATOR_PROMPT.format(
        tool_catalog=_tool_catalog(),
        order_ids=", ".join(ORDERS.keys()),
        sku_ids=", ".join(INVENTORY.keys()),
        description=description.strip(),
    )

    client = _get_client()
    response = with_retry_sync(
        lambda: client.models.generate_content(model=GENERATOR_MODEL, contents=prompt),
        attempts=4,
        base_delay=20.0,
    )
    parsed = _parse_json(response.text or "")

    expected_tools = [
        ExpectedToolCall(name=name)
        for name in parsed.get("expected_tools", [])
        if name in valid_tool_names
    ]

    return TestCase(
        id="",
        name=str(parsed.get("name", "")).strip() or "Generated test case",
        trigger_prompt=str(parsed.get("trigger_prompt", "")).strip(),
        expected_tools=expected_tools,
        golden_response=str(parsed.get("golden_response", "")).strip(),
        pass_threshold=max(0, min(100, int(parsed.get("pass_threshold", 70) or 70))),
    )
