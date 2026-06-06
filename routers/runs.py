from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from core import storage
from core.evaluator import evaluate_deterministic, evaluate_semantic, overall_pass
from core.runner import execute_prompt
from models.schemas import RunReport, TestCaseResult

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/suites/{suite_id}/run", response_model=RunReport)
async def run_suite(suite_id: str):
    suite = storage.get_suite(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")
    if not suite.test_cases:
        raise HTTPException(status_code=400, detail="Suite has no test cases to run")

    results: list[TestCaseResult] = []
    for case in suite.test_cases:
        trace = await execute_prompt(case.trigger_prompt)
        deterministic = evaluate_deterministic(case, trace)
        semantic = evaluate_semantic(case, trace)
        passed = overall_pass(case, deterministic, semantic)
        results.append(
            TestCaseResult(
                test_case_id=case.id,
                test_case_name=case.name,
                trigger_prompt=case.trigger_prompt,
                passed=passed,
                trace=trace,
                deterministic=deterministic,
                semantic=semantic,
            )
        )

    pass_count = sum(1 for r in results if r.passed)
    report = RunReport(
        id=storage.new_id("run"),
        suite_id=suite.id,
        suite_name=suite.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        pass_count=pass_count,
        fail_count=len(results) - pass_count,
        results=results,
    )
    storage.save_run(report)
    return report


@router.get("/runs/{run_id}", response_model=RunReport)
def get_run(run_id: str):
    report = storage.get_run(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Run not found")
    return report


@router.get("/runs")
def list_runs(suite_id: Optional[str] = None):
    return storage.list_runs(suite_id)
