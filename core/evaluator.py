"""Evaluation engine: combines deterministic trace assertions with an
LLM-as-judge (Gemini) semantic scoring pass.
"""
from __future__ import annotations

import json
import os
import re

from google import genai

from core.retry import with_retry_sync
from models.schemas import (
    DeterministicResult,
    RunTrace,
    SemanticResult,
    StepDiffEntry,
    TestCase,
)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _client


# ---------------------------------------------------------------------------
# Deterministic assertions: did the agent call the right tools, in order?
# ---------------------------------------------------------------------------

def evaluate_deterministic(test_case: TestCase, trace: RunTrace) -> DeterministicResult:
    expected = test_case.expected_tools
    actual = trace.steps

    diff: list[StepDiffEntry] = []
    divergence_step: int | None = None
    notes: list[str] = []

    max_len = max(len(expected), len(actual))
    for i in range(max_len):
        exp = expected[i] if i < len(expected) else None
        act = actual[i] if i < len(actual) else None

        exp_name = exp.name if exp else None
        act_name = act.tool_name if act else None
        match = exp_name is not None and act_name is not None and exp_name == act_name

        if match and exp.expected_args:
            for key, val in exp.expected_args.items():
                if act.args.get(key) != val:
                    match = False
                    notes.append(
                        f"Step {i + 1}: '{act_name}' called with {key}={act.args.get(key)!r}, "
                        f"expected {key}={val!r}"
                    )
                    break

        diff.append(StepDiffEntry(index=i + 1, expected=exp_name, actual=act_name, match=match))

        if not match and divergence_step is None:
            divergence_step = i + 1
            if exp_name and act_name:
                notes.append(f"Step {i + 1}: expected tool '{exp_name}' but agent called '{act_name}'")
            elif exp_name and not act_name:
                notes.append(f"Step {i + 1}: expected tool '{exp_name}' but agent stopped early")
            elif act_name and not exp_name:
                notes.append(f"Step {i + 1}: agent called unexpected extra tool '{act_name}'")

    passed = divergence_step is None
    return DeterministicResult(passed=passed, divergence_step=divergence_step, step_diff=diff, notes=notes)


# ---------------------------------------------------------------------------
# Semantic evaluation: LLM-as-judge via Gemini
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are an impartial evaluator grading an AI customer-support agent's response.

User request:
{prompt}

Golden (reference) response — what an ideal agent would say:
{golden}

Agent's actual response:
{actual}

Score how well the agent's actual response matches the intent, key facts, and
completeness of the golden response on a scale from 0 to 100, where:
- 90-100: Matches intent and all key facts, no fabricated information
- 70-89: Matches intent and most key facts, minor omissions
- 40-69: Partially matches intent, missing important facts or somewhat off-topic
- 0-39: Wrong intent, missing critical facts, or contains fabricated/hallucinated information

Also flag whether the actual response appears to hallucinate facts not supported
by the golden response or the conversation (true/false).

Respond with ONLY a JSON object in this exact shape, no markdown fences:
{{"score": <integer 0-100>, "hallucination": <true|false>, "rationale": "<one or two sentence explanation>"}}
"""


def _parse_judge_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def evaluate_semantic(test_case: TestCase, trace: RunTrace) -> SemanticResult:
    if not test_case.golden_response.strip():
        # No golden response provided — nothing meaningful to grade semantically.
        return SemanticResult(score=100, rationale="No golden response provided; semantic check skipped.", hallucination_flag=False)

    prompt = _JUDGE_PROMPT.format(
        prompt=test_case.trigger_prompt,
        golden=test_case.golden_response,
        actual=trace.final_response or "(agent produced no final response)",
    )

    try:
        client = _get_client()
        response = with_retry_sync(
            lambda: client.models.generate_content(model=JUDGE_MODEL, contents=prompt),
            attempts=4,
            base_delay=20.0,
        )
        parsed = _parse_judge_json(response.text or "")
        score = max(0, min(100, int(parsed.get("score", 0))))
        return SemanticResult(
            score=score,
            rationale=str(parsed.get("rationale", "")).strip(),
            hallucination_flag=bool(parsed.get("hallucination", False)),
        )
    except Exception as exc:  # noqa: BLE001 - judge failures shouldn't crash the run
        return SemanticResult(
            score=0,
            rationale=f"Judge evaluation failed: {exc}",
            hallucination_flag=False,
        )


def overall_pass(test_case: TestCase, deterministic: DeterministicResult, semantic: SemanticResult) -> bool:
    return deterministic.passed and semantic.score >= test_case.pass_threshold
