from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.db.database import Database
from app.db.repositories import CandidateProfileRepository, RunRepository
from app.models.schemas import (
    CandidateProfile,
    MessageType,
    RecommendedAction,
    ReviewDecision,
    RunRecord,
    RunStatus,
    UrgencyLevel,
)
from app.services.exceptions import InvalidRunTransitionError


def test_candidate_profile_repository_seeds_default_profile(tmp_path: Path) -> None:
    database = Database(tmp_path / "signaldraft.db")
    database.initialize()
    repository = CandidateProfileRepository(database)
    profile = repository.get()
    assert profile.full_name == "Alex Morgan"
    assert profile.preferred_tone == "warm"


def test_run_repository_round_trip(tmp_path: Path) -> None:
    database = Database(tmp_path / "signaldraft.db")
    database.initialize()
    repository = RunRepository(database)
    now = datetime.utcnow()
    run = RunRecord(
        run_id="run-123",
        raw_message="Hello there",
        normalized_message="Hello there",
        message_type=MessageType.unknown,
        urgency=UrgencyLevel.low,
        extracted={"company": "Nimbus AI"},
        candidate_profile=CandidateProfile().model_dump(mode="json"),
        recommended_action=RecommendedAction.draft_reply,
        draft_reply="Thanks for reaching out.",
        needs_human_review=False,
        review_reason="",
        explanation="Looks safe to reply.",
        workflow_steps=[],
        errors=[],
        status=RunStatus.analyzed,
        created_at=now,
        updated_at=now,
    )
    repository.save(run)
    loaded = repository.get("run-123")
    assert loaded is not None
    assert loaded.run_id == "run-123"
    assert loaded.extracted["company"] == "Nimbus AI"


def test_mock_send_requires_approval(tmp_path: Path) -> None:
    database = Database(tmp_path / "signaldraft.db")
    database.initialize()
    repository = RunRepository(database)
    now = datetime.utcnow()
    run = RunRecord(
        run_id="run-approval",
        raw_message="Hello there",
        normalized_message="Hello there",
        message_type=MessageType.recruiter_outreach,
        urgency=UrgencyLevel.low,
        extracted={},
        candidate_profile=CandidateProfile().model_dump(mode="json"),
        recommended_action=RecommendedAction.draft_reply,
        draft_reply="Thanks for reaching out.",
        needs_human_review=False,
        review_reason="",
        explanation="Looks safe to reply.",
        workflow_steps=[],
        errors=[],
        status=RunStatus.analyzed,
        created_at=now,
        updated_at=now,
    )
    repository.save(run)

    with pytest.raises(InvalidRunTransitionError, match="Only approved runs can be mock sent"):
        repository.mark_mock_sent("run-approval")


def test_archived_run_cannot_be_reviewed_or_sent(tmp_path: Path) -> None:
    database = Database(tmp_path / "signaldraft.db")
    database.initialize()
    repository = RunRepository(database)
    now = datetime.utcnow()
    run = RunRecord(
        run_id="run-archived",
        raw_message="Spam",
        normalized_message="Spam",
        message_type=MessageType.spam_or_low_value,
        urgency=UrgencyLevel.low,
        extracted={},
        candidate_profile=CandidateProfile().model_dump(mode="json"),
        recommended_action=RecommendedAction.archive_no_reply,
        draft_reply="",
        needs_human_review=False,
        review_reason="",
        explanation="Archive the message.",
        workflow_steps=[],
        errors=[],
        status=RunStatus.analyzed,
        created_at=now,
        updated_at=now,
    )
    repository.save(run)

    with pytest.raises(InvalidRunTransitionError, match="Archived runs cannot be approved or rejected"):
        repository.save_feedback("run-archived", ReviewDecision.approved, None, None)

    with pytest.raises(InvalidRunTransitionError, match="Archived runs cannot be mock sent"):
        repository.mark_mock_sent("run-archived")
