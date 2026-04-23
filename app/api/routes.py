from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.auth import require_api_token
from app.models.schemas import (
    AnalyzeRequest,
    CandidateProfile,
    HealthResponse,
    MockSendRequest,
    ReadinessResponse,
    ReviewRequest,
    RunRecord,
    RunsResponse,
)
from app.services.container import container
from app.services.exceptions import InvalidRunTransitionError
from app.utils.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        backend_auth_enabled=settings.backend_auth_enabled,
    )


@router.get("/readiness", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    return container.readiness_status()


@router.post("/analyze", response_model=RunRecord, dependencies=[Depends(require_api_token)])
def analyze(request: AnalyzeRequest) -> RunRecord:
    return container.analysis_service.analyze_message(request.raw_message)


@router.get("/runs", response_model=RunsResponse, dependencies=[Depends(require_api_token)])
def list_runs(limit: int = Query(default=25, ge=1, le=100)) -> RunsResponse:
    return container.analysis_service.list_runs(limit=limit)


@router.get("/runs/{run_id}", response_model=RunRecord, dependencies=[Depends(require_api_token)])
def get_run(run_id: str) -> RunRecord:
    run = container.analysis_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/review", response_model=RunRecord, dependencies=[Depends(require_api_token)])
def review_run(run_id: str, request: ReviewRequest) -> RunRecord:
    try:
        run = container.analysis_service.review_run(
            run_id=run_id,
            decision=request.decision,
            edited_draft=request.edited_draft,
            notes=request.notes,
        )
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run
    except InvalidRunTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@router.post("/runs/{run_id}/mock-send", response_model=RunRecord, dependencies=[Depends(require_api_token)])
def mock_send(run_id: str, request: MockSendRequest) -> RunRecord:
    try:
        run = container.analysis_service.mock_send(run_id, request.edited_draft)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run
    except InvalidRunTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@router.get("/profile", response_model=CandidateProfile, dependencies=[Depends(require_api_token)])
def get_profile() -> CandidateProfile:
    return container.analysis_service.get_profile()


@router.put("/profile", response_model=CandidateProfile, dependencies=[Depends(require_api_token)])
def save_profile(profile: CandidateProfile) -> CandidateProfile:
    return container.analysis_service.save_profile(profile)
