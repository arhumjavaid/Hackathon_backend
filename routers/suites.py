from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core import storage
from core.generator import generate_test_case
from models.schemas import GenerateTestCaseRequest, TestCase, TestSuite, TestSuiteCreate

router = APIRouter(prefix="/api/suites", tags=["suites"])


@router.post("/generate-test-case", response_model=TestCase)
def generate_test_case_draft(payload: GenerateTestCaseRequest):
    if not payload.description.strip():
        raise HTTPException(status_code=400, detail="Description is required")
    try:
        return generate_test_case(payload.description)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to generate test case: {exc}")


@router.get("", response_model=list[TestSuite])
def list_suites():
    return storage.list_suites()


@router.post("", response_model=TestSuite)
def create_suite(payload: TestSuiteCreate):
    suite = TestSuite(
        id=storage.new_id("suite"),
        name=payload.name,
        description=payload.description,
        test_cases=[
            tc if tc.id else TestCase(**{**tc.model_dump(), "id": storage.new_id("case")})
            for tc in payload.test_cases
        ],
    )
    # Ensure every test case has an id even if the client sent empty strings
    for tc in suite.test_cases:
        if not tc.id:
            tc.id = storage.new_id("case")
    return storage.create_suite(suite)


@router.get("/{suite_id}", response_model=TestSuite)
def get_suite(suite_id: str):
    suite = storage.get_suite(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")
    return suite


@router.put("/{suite_id}", response_model=TestSuite)
def update_suite(suite_id: str, payload: TestSuiteCreate):
    existing = storage.get_suite(suite_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Suite not found")
    test_cases = []
    for tc in payload.test_cases:
        data = tc.model_dump()
        if not data.get("id"):
            data["id"] = storage.new_id("case")
        test_cases.append(TestCase(**data))
    updated = TestSuite(id=suite_id, name=payload.name, description=payload.description, test_cases=test_cases)
    storage.update_suite(suite_id, updated)
    return updated


@router.delete("/{suite_id}")
def delete_suite(suite_id: str):
    if not storage.delete_suite(suite_id):
        raise HTTPException(status_code=404, detail="Suite not found")
    return {"deleted": True}
