from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    recruiter_outreach = "recruiter_outreach"
    interview_invite = "interview_invite"
    scheduling_request = "scheduling_request"
    coding_assessment = "coding_assessment"
    rejection = "rejection"
    offer_related = "offer_related"
    networking_reply = "networking_reply"
    follow_up_needed = "follow_up_needed"
    spam_or_low_value = "spam_or_low_value"
    unknown = "unknown"


class UrgencyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RecommendedAction(str, Enum):
    draft_reply = "draft_reply"
    ask_for_missing_info = "ask_for_missing_info"
    archive_no_reply = "archive_no_reply"
    escalate_human_review = "escalate_human_review"


class WorkflowStatus(str, Enum):
    completed = "completed"
    warning = "warning"
    skipped = "skipped"


class ReviewDecision(str, Enum):
    approved = "approved"
    rejected = "rejected"


class RunStatus(str, Enum):
    analyzed = "analyzed"
    approved = "approved"
    rejected = "rejected"
    mock_sent = "mock_sent"


class WorkflowStep(BaseModel):
    name: str
    status: WorkflowStatus
    summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateProfile(BaseModel):
    full_name: str = "Alex Morgan"
    email: str = "alex@example.com"
    university: str = "McMaster University"
    graduation_date: str = "2026-05"
    resume_summary: str = (
        "Full-stack software engineer focused on AI products, automation, and polished user experiences."
    )
    preferred_tone: Literal["formal", "warm", "concise"] = "warm"
    target_roles: list[str] = Field(default_factory=lambda: ["AI Engineer", "Full-Stack Engineer"])
    location: str = "Toronto, ON"
    sponsorship_status: str = "Requires visa sponsorship in the United States."
    portfolio_links: list[str] = Field(default_factory=lambda: ["https://github.com/example", "https://example.dev"])
    calendar_preferences: str = "Weekdays 9am-5pm ET, prefer 24-hour notice for interviews."
    default_signoff: str = "Best regards"


class ClassificationOutput(BaseModel):
    message_type: MessageType
    urgency: UrgencyLevel
    confidence: float = Field(ge=0.0, le=1.0, default=0.75)
    rationale: str = Field(default="")


class ExtractionOutput(BaseModel):
    sender_name: str | None = None
    sender_email: str | None = None
    company: str | None = None
    role_title: str | None = None
    message_type: MessageType | None = None
    urgency: UrgencyLevel | None = None
    deadline: str | None = None
    timezone: str | None = None
    requested_action: str | None = None
    interview_stage: str | None = None
    compensation_mentioned: bool = False
    sponsorship_mentioned: bool = False
    meeting_link_present: bool = False
    dates_mentioned: list[str] = Field(default_factory=list)
    next_step_summary: str | None = None


class DecisionOutput(BaseModel):
    recommended_action: RecommendedAction
    needs_human_review: bool = False
    review_reason: str = ""
    missing_information: list[str] = Field(default_factory=list)
    explanation: str = ""


class SafetyReviewOutput(BaseModel):
    approved: bool = True
    needs_human_review: bool = False
    risk_note: str = ""
    revised_draft: str = ""
    explanation: str = ""


class AnalyzeRequest(BaseModel):
    raw_message: str = Field(min_length=1)


class ReviewRequest(BaseModel):
    decision: ReviewDecision
    edited_draft: str | None = None
    notes: str | None = None


class MockSendRequest(BaseModel):
    edited_draft: str | None = None


class RunRecord(BaseModel):
    run_id: str
    raw_message: str
    normalized_message: str
    message_type: MessageType
    urgency: UrgencyLevel
    extracted: dict[str, Any]
    candidate_profile: dict[str, Any]
    recommended_action: RecommendedAction
    draft_reply: str
    needs_human_review: bool
    review_reason: str
    explanation: str
    workflow_steps: list[WorkflowStep] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.analyzed
    review_notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RunSummary(BaseModel):
    run_id: str
    message_type: MessageType
    urgency: UrgencyLevel
    recommended_action: RecommendedAction
    needs_human_review: bool
    status: RunStatus
    created_at: datetime
    preview: str


class RunsResponse(BaseModel):
    items: list[RunSummary]


class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_mode: str
    openai_configured: bool
    langsmith_project: str
