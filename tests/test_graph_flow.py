from __future__ import annotations

from pathlib import Path

from app.db.database import Database
from app.db.repositories import CandidateProfileRepository, RunRepository
from app.graph.builder import SignalDraftGraph
from app.models.schemas import ClassificationOutput, ExtractionOutput, MessageType, UrgencyLevel
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMService
from app.utils.config import Settings


def test_graph_routes_to_human_review_for_sensitive_message(tmp_path: Path) -> None:
    settings = Settings(
        llm_mode="heuristic",
        fallback_to_rules=True,
        db_path=tmp_path / "runs.db",
        checkpoint_path=tmp_path / "checkpoints.db",
    )
    database = Database(settings.db_path)
    database.initialize()
    profile_repository = CandidateProfileRepository(database)
    run_repository = RunRepository(database)
    llm_service = LLMService(settings)
    graph = SignalDraftGraph(llm_service, profile_repository, settings.checkpoint_path)
    service = AnalysisService(graph, llm_service, profile_repository, run_repository)

    run = service.analyze_message(
        "Hi Alex, can you confirm whether you need visa sponsorship and your salary expectations?"
    )

    assert run.needs_human_review is True
    assert run.recommended_action.value == "escalate_human_review"
    assert any(step.name == "decide_action" for step in run.workflow_steps)

    graph.close()


class MisclassifyingLLMService:
    model_available = True

    @staticmethod
    def classify_message(_message: str) -> ClassificationOutput:
        return ClassificationOutput(message_type=MessageType.recruiter_outreach, urgency=UrgencyLevel.medium)

    @staticmethod
    def extract_fields(
        message: str,
        message_type: str,
        urgency: str,
        fallback_classification: ClassificationOutput,
    ) -> ExtractionOutput:
        return ExtractionOutput(
            sender_name="Megan Ortiz",
            sender_email="megan@atlascommerce.com",
            company="Atlas Commerce",
            role_title="Senior ML Engineer",
            message_type=MessageType.recruiter_outreach,
            urgency=UrgencyLevel.medium,
            requested_action="Confirm visa sponsorship requirement and share compensation expectations",
            compensation_mentioned=True,
            sponsorship_mentioned=True,
            next_step_summary="Confirm visa sponsorship requirement and share compensation expectations",
        )

    @staticmethod
    def assist_decision(message: str, extracted: dict[str, object]):
        return None

    @staticmethod
    def draft_response(*args, **kwargs):
        return ""

    @staticmethod
    def review_draft(message: str, extracted: dict[str, object], draft_reply: str):
        raise AssertionError("Safety review should not run for escalated messages")


def test_graph_normalizes_sensitive_recruiter_message_to_offer_related(tmp_path: Path) -> None:
    settings = Settings(
        llm_mode="heuristic",
        fallback_to_rules=True,
        db_path=tmp_path / "runs.db",
        checkpoint_path=tmp_path / "checkpoints.db",
    )
    database = Database(settings.db_path)
    database.initialize()
    profile_repository = CandidateProfileRepository(database)
    run_repository = RunRepository(database)
    llm_service = MisclassifyingLLMService()
    graph = SignalDraftGraph(llm_service, profile_repository, settings.checkpoint_path)
    service = AnalysisService(graph, llm_service, profile_repository, run_repository)

    run = service.analyze_message(
        "Hi Alex, we are excited about your profile for the Senior ML Engineer opportunity at Atlas Commerce. "
        "Before we move ahead, can you confirm whether you require visa sponsorship and share your compensation "
        "expectations for a base salary range of $170,000-$190,000?"
    )

    assert run.message_type.value == "offer_related"
    assert run.recommended_action.value == "escalate_human_review"
    assert any("normalized message type to offer_related" in step.summary for step in run.workflow_steps)

    graph.close()
