"""The 'agent under test': a customer-support / order-management ADK agent
wired up with 6 tools and a Gemini model.
"""
from __future__ import annotations

import os

from google.adk.agents import Agent

from agent.tools import ALL_TOOLS

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-2.5-flash")

INSTRUCTION = """You are a customer-support assistant for an online electronics store.

You can look up orders, check product inventory, check refund eligibility,
process refunds, send confirmation emails, and escalate to a human agent.

Guidelines:
- Always look up the order before taking any action on it.
- Before processing a refund, you MUST check the refund policy/eligibility for that order.
  Never process a refund for an order that is not eligible — escalate to a human instead.
- After successfully completing an action for the customer (refund, etc.), send a
  confirmation email summarizing what was done.
- If a request can't be resolved with your tools or policy allows, escalate to a human.
- Be concise and clear in your final response to the customer.
"""

root_agent = Agent(
    name="support_agent",
    model=AGENT_MODEL,
    description="Customer support agent for order lookups, refunds, and escalations.",
    instruction=INSTRUCTION,
    tools=ALL_TOOLS,
)


def build_agent() -> Agent:
    """Factory so the runner can create a fresh agent instance per run if needed."""
    return Agent(
        name="support_agent",
        model=AGENT_MODEL,
        description="Customer support agent for order lookups, refunds, and escalations.",
        instruction=INSTRUCTION,
        tools=ALL_TOOLS,
    )
