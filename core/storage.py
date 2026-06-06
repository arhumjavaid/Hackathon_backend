"""Simple JSON-file persistence for test suites and run reports."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from models.schemas import RunReport, RunSummary, TestSuite

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUITES_FILE = DATA_DIR / "suites.json"
RUNS_DIR = DATA_DIR / "runs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
if not SUITES_FILE.exists():
    SUITES_FILE.write_text("[]")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def list_suites() -> list[TestSuite]:
    raw = json.loads(SUITES_FILE.read_text())
    return [TestSuite(**s) for s in raw]


def get_suite(suite_id: str) -> Optional[TestSuite]:
    for suite in list_suites():
        if suite.id == suite_id:
            return suite
    return None


def save_suites(suites: list[TestSuite]) -> None:
    SUITES_FILE.write_text(json.dumps([s.model_dump() for s in suites], indent=2))


def create_suite(suite: TestSuite) -> TestSuite:
    suites = list_suites()
    suites.append(suite)
    save_suites(suites)
    return suite


def update_suite(suite_id: str, updated: TestSuite) -> Optional[TestSuite]:
    suites = list_suites()
    for i, suite in enumerate(suites):
        if suite.id == suite_id:
            suites[i] = updated
            save_suites(suites)
            return updated
    return None


def delete_suite(suite_id: str) -> bool:
    suites = list_suites()
    remaining = [s for s in suites if s.id != suite_id]
    if len(remaining) == len(suites):
        return False
    save_suites(remaining)
    return True


# ---------------------------------------------------------------------------
# Run reports
# ---------------------------------------------------------------------------

def save_run(report: RunReport) -> RunReport:
    path = RUNS_DIR / f"{report.id}.json"
    path.write_text(json.dumps(report.model_dump(), indent=2))
    return report


def get_run(run_id: str) -> Optional[RunReport]:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return RunReport(**json.loads(path.read_text()))


def list_runs(suite_id: Optional[str] = None) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            report = RunReport(**json.loads(path.read_text()))
        except Exception:
            continue
        if suite_id and report.suite_id != suite_id:
            continue
        scores = [r.semantic.score for r in report.results]
        tokens = [r.trace.token_usage.total_tokens for r in report.results]
        latencies = [r.trace.latency_ms for r in report.results]
        summaries.append(
            RunSummary(
                id=report.id,
                suite_id=report.suite_id,
                created_at=report.created_at,
                pass_count=report.pass_count,
                fail_count=report.fail_count,
                avg_semantic_score=round(sum(scores) / len(scores), 1) if scores else 0,
                total_tokens=sum(tokens),
                avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0,
            )
        )
    summaries.sort(key=lambda s: s.created_at)
    return summaries
