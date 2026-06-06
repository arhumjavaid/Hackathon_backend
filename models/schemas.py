"""Pydantic models shared across the API, runner, and evaluator."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Test suite / test case definitions (authored via the Suite Builder UI)
# ---------------------------------------------------------------------------

class ExpectedToolCall(BaseModel):
    name: str
    expected_args: Optional[dict[str, Any]] = None


class TestCase(BaseModel):
    id: str
    name: str
    trigger_prompt: str
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
    golden_response: str = ""
    pass_threshold: int = 70


class TestSuite(BaseModel):
    id: str
    name: str
    description: str = ""
    test_cases: list[TestCase] = Field(default_factory=list)


class TestSuiteCreate(BaseModel):
    name: str
    description: str = ""
    test_cases: list[TestCase] = Field(default_factory=list)


class GenerateTestCaseRequest(BaseModel):
    description: str


# ---------------------------------------------------------------------------
# Execution trace captured by the runner
# ---------------------------------------------------------------------------

class ToolCallStep(BaseModel):
    step: int
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0


class RunTrace(BaseModel):
    steps: list[ToolCallStep] = Field(default_factory=list)
    final_response: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class StepDiffEntry(BaseModel):
    index: int
    expected: Optional[str] = None
    actual: Optional[str] = None
    match: bool


class DeterministicResult(BaseModel):
    passed: bool
    divergence_step: Optional[int] = None
    step_diff: list[StepDiffEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SemanticResult(BaseModel):
    score: int
    rationale: str
    hallucination_flag: bool = False


class TestCaseResult(BaseModel):
    test_case_id: str
    test_case_name: str
    trigger_prompt: str
    passed: bool
    trace: RunTrace
    deterministic: DeterministicResult
    semantic: SemanticResult


class RunReport(BaseModel):
    id: str
    suite_id: str
    suite_name: str
    created_at: str
    pass_count: int
    fail_count: int
    results: list[TestCaseResult] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Lightweight entry used for suite history / trend charts."""
    id: str
    suite_id: str
    created_at: str
    pass_count: int
    fail_count: int
    avg_semantic_score: float
    total_tokens: int
    avg_latency_ms: float
