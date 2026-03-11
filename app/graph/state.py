from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    raw_message: str
    normalized_message: str
    message_type: str
    urgency: str
    extracted: dict[str, Any]
    candidate_profile: dict[str, Any]
    recommended_action: str
    draft_reply: str
    needs_human_review: bool
    review_reason: str
    explanation: str
    errors: list[str]
    workflow_steps: list[dict[str, Any]]
    missing_information: list[str]

