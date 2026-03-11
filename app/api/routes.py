from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    AnalyzeRequest,
    CandidateProfile,
    HealthResponse,
    MockSendRequest,
    ReviewRequest,
    RunRecord,
    RunsResponse,
)
from app.services.container import container
from app.utils.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        llm_mode=settings.llm_mode,
        openai_configured=bool(settings.openai_api_key),
        langsmith_project=settings.langsmith_project,
    )


@router.post("/analyze", response_model=RunRecord)
def analyze(request: AnalyzeRequest) -> RunRecord:
    return container.analysis_service.analyze_message(request.raw_message)


@router.get("/runs", response_model=RunsResponse)
def list_runs(limit: int = Query(default=25, ge=1, le=100)) -> RunsResponse:
    return container.analysis_service.list_runs(limit=limit)


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    run = container.analysis_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/review", response_model=RunRecord)
def review_run(run_id: str, request: ReviewRequest) -> RunRecord:
    run = container.analysis_service.review_run(
        run_id=run_id,
        decision=request.decision,
        edited_draft=request.edited_draft,
        notes=request.notes,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/mock-send", response_model=RunRecord)
def mock_send(run_id: str, request: MockSendRequest) -> RunRecord:
    run = container.analysis_service.mock_send(run_id, request.edited_draft)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/profile", response_model=CandidateProfile)
def get_profile() -> CandidateProfile:
    return container.analysis_service.get_profile()


@router.put("/profile", response_model=CandidateProfile)
def save_profile(profile: CandidateProfile) -> CandidateProfile:
    return container.analysis_service.save_profile(profile)

