from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.db.repositories import CandidateProfileRepository, RunRepository
from app.graph.builder import SignalDraftGraph
from app.graph.state import AgentState
from app.models.schemas import CandidateProfile, MessageType, RunRecord, RunStatus, UrgencyLevel, WorkflowStep
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisService:
    def __init__(
        self,
        graph: SignalDraftGraph,
        profile_repository: CandidateProfileRepository,
        run_repository: RunRepository,
    ) -> None:
        self.graph = graph
        self.profile_repository = profile_repository
        self.run_repository = run_repository

    def analyze_message(self, raw_message: str) -> RunRecord:
        run_id = str(uuid4())
        logger.info("Starting SignalDraft run %s", run_id)
        initial_state: AgentState = {
            "run_id": run_id,
            "raw_message": raw_message,
            "errors": [],
            "workflow_steps": [],
            "draft_reply": "",
            "review_reason": "",
            "explanation": "",
        }
        result_state = self.graph.invoke(initial_state, run_id=run_id)
        run = self._state_to_record(result_state)
        self.run_repository.save(run)
        logger.info("Completed SignalDraft run %s", run_id)
        return run

    def list_runs(self, limit: int = 25):
        return self.run_repository.list_runs(limit=limit)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.run_repository.get(run_id)

    def get_profile(self) -> CandidateProfile:
        return self.profile_repository.get()

    def save_profile(self, profile: CandidateProfile) -> CandidateProfile:
        return self.profile_repository.save(profile)

    def review_run(self, run_id: str, decision, edited_draft: str | None, notes: str | None) -> RunRecord | None:
        return self.run_repository.save_feedback(run_id, decision, edited_draft, notes)

    def mock_send(self, run_id: str, edited_draft: str | None = None) -> RunRecord | None:
        return self.run_repository.mark_mock_sent(run_id, edited_draft)

    @staticmethod
    def _state_to_record(state: AgentState) -> RunRecord:
        now = datetime.utcnow()
        return RunRecord(
            run_id=state["run_id"],
            raw_message=state["raw_message"],
            normalized_message=state.get("normalized_message", state["raw_message"]),
            message_type=MessageType(state.get("message_type", MessageType.unknown.value)),
            urgency=UrgencyLevel(state.get("urgency", UrgencyLevel.low.value)),
            extracted=state.get("extracted", {}),
            candidate_profile=state.get("candidate_profile", {}),
            recommended_action=state.get("recommended_action", "draft_reply"),
            draft_reply=state.get("draft_reply", ""),
            needs_human_review=bool(state.get("needs_human_review", False)),
            review_reason=state.get("review_reason", ""),
            explanation=state.get("explanation", ""),
            workflow_steps=[WorkflowStep.model_validate(step) for step in state.get("workflow_steps", [])],
            errors=list(state.get("errors", [])),
            status=RunStatus.analyzed,
            created_at=now,
            updated_at=now,
        )

