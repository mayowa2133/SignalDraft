from __future__ import annotations

from datetime import datetime

from app.db.database import Database
from app.models.schemas import CandidateProfile, ReviewDecision, RunRecord, RunStatus, RunSummary, RunsResponse
from app.services.exceptions import InvalidRunTransitionError


class CandidateProfileRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def get(self) -> CandidateProfile:
        with self.database.connection() as conn:
            row = conn.execute("SELECT payload FROM candidate_profile WHERE id = 1").fetchone()
            if row is None:
                profile = CandidateProfile()
                self.save(profile)
                return profile
            return CandidateProfile.model_validate_json(row["payload"])

    def save(self, profile: CandidateProfile) -> CandidateProfile:
        timestamp = datetime.utcnow().isoformat()
        payload = profile.model_dump_json()
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO candidate_profile (id, payload, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (payload, timestamp),
            )
        return profile


class RunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, run: RunRecord) -> RunRecord:
        payload = run.model_dump_json()
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (run.run_id, payload, run.created_at.isoformat(), run.updated_at.isoformat()),
            )
        return run

    def get(self, run_id: str) -> RunRecord | None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            return RunRecord.model_validate_json(row["payload"])

    def list_runs(self, limit: int = 25) -> RunsResponse:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items = [self._to_summary(RunRecord.model_validate_json(row["payload"])) for row in rows]
        return RunsResponse(items=items)

    def save_feedback(
        self,
        run_id: str,
        decision: ReviewDecision,
        edited_draft: str | None,
        notes: str | None,
    ) -> RunRecord | None:
        run = self.get(run_id)
        if run is None:
            return None
        self._validate_review_transition(run)
        timestamp = datetime.utcnow()
        if edited_draft is not None:
            run.draft_reply = edited_draft
        if notes:
            run.review_notes = notes
        run.status = RunStatus.approved if decision == ReviewDecision.approved else RunStatus.rejected
        run.updated_at = timestamp
        self.save(run)
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO run_feedback (run_id, decision, edited_draft, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, decision.value, edited_draft, notes, timestamp.isoformat()),
            )
        return run

    def mark_mock_sent(self, run_id: str, edited_draft: str | None = None) -> RunRecord | None:
        run = self.get(run_id)
        if run is None:
            return None
        self._validate_mock_send_transition(run)
        if edited_draft is not None:
            run.draft_reply = edited_draft
        run.status = RunStatus.mock_sent
        run.updated_at = datetime.utcnow()
        self.save(run)
        return run

    @staticmethod
    def _validate_review_transition(run: RunRecord) -> None:
        if run.recommended_action.value == "archive_no_reply":
            raise InvalidRunTransitionError(
                code="review_not_allowed",
                message="Archived runs cannot be approved or rejected.",
            )
        if run.status != RunStatus.analyzed:
            raise InvalidRunTransitionError(
                code="invalid_review_transition",
                message=f"Only analyzed runs can be reviewed. Current status: {run.status.value}.",
            )

    @staticmethod
    def _validate_mock_send_transition(run: RunRecord) -> None:
        if run.recommended_action.value == "archive_no_reply":
            raise InvalidRunTransitionError(
                code="mock_send_not_allowed",
                message="Archived runs cannot be mock sent.",
            )
        if run.status != RunStatus.approved:
            raise InvalidRunTransitionError(
                code="mock_send_requires_approval",
                message=f"Only approved runs can be mock sent. Current status: {run.status.value}.",
            )

    @staticmethod
    def _to_summary(run: RunRecord) -> RunSummary:
        preview = run.raw_message.replace("\n", " ").strip()
        if len(preview) > 92:
            preview = f"{preview[:89]}..."
        return RunSummary(
            run_id=run.run_id,
            message_type=run.message_type,
            urgency=run.urgency,
            recommended_action=run.recommended_action,
            needs_human_review=run.needs_human_review,
            status=run.status,
            created_at=run.created_at,
            preview=preview,
        )
