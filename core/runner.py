"""Execution Runner: drives the ADK agent in-process for a single test case
and captures a structured trace of everything that happened — tool calls
(with args + results), the final response, token usage, and latency.
"""
from __future__ import annotations

import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent.sample_agent import build_agent
from core.retry import with_retry_async
from models.schemas import RunTrace, ToolCallStep, TokenUsage

APP_NAME = "agent_test_harness"
USER_ID = "test_runner"


async def execute_prompt(trigger_prompt: str) -> RunTrace:
    """Run a single prompt through a fresh agent + session and capture the trace.

    Wrapped in retry-with-backoff since free-tier Gemini keys have tight
    per-minute quotas and 429s are common when running several test cases back to back.
    """
    return await with_retry_async(lambda: _run_once(trigger_prompt), attempts=4, base_delay=20.0)


async def _run_once(trigger_prompt: str) -> RunTrace:
    agent = build_agent()
    session_service = InMemorySessionService()
    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

    message = types.Content(role="user", parts=[types.Part(text=trigger_prompt)])

    steps: list[ToolCallStep] = []
    pending_calls: dict[str, dict] = {}  # function_call.id -> {tool_name, args, step}
    final_response = ""
    usage = TokenUsage()
    step_counter = 0

    start = time.perf_counter()
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
        if event.usage_metadata:
            usage.prompt_tokens = max(usage.prompt_tokens, event.usage_metadata.prompt_token_count or 0)
            usage.candidate_tokens += event.usage_metadata.candidates_token_count or 0
            usage.total_tokens = max(usage.total_tokens, event.usage_metadata.total_token_count or 0)

        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if part.function_call:
                step_counter += 1
                fc = part.function_call
                call_id = fc.id or f"call_{step_counter}"
                pending_calls[call_id] = {
                    "tool_name": fc.name,
                    "args": dict(fc.args or {}),
                    "step": step_counter,
                }
            elif part.function_response:
                fr = part.function_response
                call_id = fr.id or ""
                pending = pending_calls.pop(call_id, None)
                if pending is None:
                    # Fall back to matching by name if id wasn't propagated
                    pending = {"tool_name": fr.name, "args": {}, "step": len(steps) + 1}
                steps.append(
                    ToolCallStep(
                        step=pending["step"],
                        tool_name=pending["tool_name"],
                        args=pending["args"],
                        result=fr.response,
                    )
                )
            elif part.text and not part.thought:
                final_response += part.text

    latency_ms = (time.perf_counter() - start) * 1000

    steps.sort(key=lambda s: s.step)
    return RunTrace(
        steps=steps,
        final_response=final_response.strip(),
        token_usage=usage,
        latency_ms=round(latency_ms, 1),
    )
