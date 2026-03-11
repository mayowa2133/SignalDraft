from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from app.models.schemas import (
    CandidateProfile,
    ClassificationOutput,
    DecisionOutput,
    ExtractionOutput,
    MessageType,
    RecommendedAction,
    UrgencyLevel,
    WorkflowStatus,
)


MESSAGE_KEYWORDS: list[tuple[MessageType, tuple[str, ...]]] = [
    (MessageType.offer_related, ("offer", "compensation", "equity", "base salary")),
    (MessageType.coding_assessment, ("coding challenge", "assessment", "hackerrank", "take-home")),
    (MessageType.rejection, ("unfortunately", "moving forward with other candidates", "other candidates", "regret to inform")),
    (MessageType.interview_invite, ("interview", "onsite", "technical screen", "phone screen")),
    (MessageType.scheduling_request, ("availability", "schedule", "calendar", "reschedule", "meet")),
    (MessageType.follow_up_needed, ("following up", "just checking in", "checking in", "circling back")),
    (MessageType.recruiter_outreach, ("recruiter", "opportunity", "role", "opening", "hiring")),
    (MessageType.networking_reply, ("coffee chat", "connect", "network", "intro")),
    (MessageType.spam_or_low_value, ("unsubscribe", "marketing", "newsletter", "promotional")),
]

DATE_PATTERNS = [
    r"\b(?:today|tomorrow|next week|this week)\b",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s+\d{4})?\b",
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
]

TIMEZONE_PATTERN = re.compile(
    r"\b(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC|GMT|CET|IST)\b",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
URL_PATTERN = re.compile(r"https?://\S+")
COMPENSATION_PATTERN = re.compile(r"\$[\d,]+|\b(?:salary|compensation|equity|bonus|pay range)\b", re.IGNORECASE)
SPONSORSHIP_PATTERN = re.compile(r"\b(?:visa|sponsorship|work authorization|h-1b|opt|cpt)\b", re.IGNORECASE)


def normalize_message(raw_message: str) -> str:
    return re.sub(r"\s+", " ", raw_message).strip()


def classify_with_rules(message: str) -> ClassificationOutput:
    lowered = message.lower()
    chosen = MessageType.unknown
    best_score = 0
    for message_type, keywords in MESSAGE_KEYWORDS:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            chosen = message_type
    urgency = infer_urgency_from_text(message)
    rationale = "Keyword-based fallback classification."
    confidence = 0.45 if chosen == MessageType.unknown else min(0.85, 0.55 + best_score * 0.1)
    return ClassificationOutput(message_type=chosen, urgency=urgency, confidence=confidence, rationale=rationale)


def extract_with_rules(message: str, classification: ClassificationOutput | None = None) -> ExtractionOutput:
    message_type = classification.message_type if classification else MessageType.unknown
    urgency = classification.urgency if classification else infer_urgency_from_text(message)
    sender_email_match = EMAIL_PATTERN.search(message)
    dates = extract_dates(message)
    urls = URL_PATTERN.findall(message)
    timezone_match = TIMEZONE_PATTERN.search(message)

    company = extract_company(message)
    role_title = extract_role_title(message)
    sender_name = extract_sender_name(message)
    deadline = find_deadline(message, dates)
    interview_stage = extract_interview_stage(message)

    extraction = ExtractionOutput(
        sender_name=sender_name,
        sender_email=sender_email_match.group(0) if sender_email_match else None,
        company=company,
        role_title=role_title,
        message_type=message_type,
        urgency=urgency,
        deadline=deadline,
        timezone=timezone_match.group(0).upper() if timezone_match else None,
        requested_action=extract_requested_action(message),
        interview_stage=interview_stage,
        compensation_mentioned=bool(COMPENSATION_PATTERN.search(message)),
        sponsorship_mentioned=bool(SPONSORSHIP_PATTERN.search(message)),
        meeting_link_present=any(urlparse(url).scheme in {"http", "https"} for url in urls),
        dates_mentioned=dates,
        next_step_summary=extract_next_step_summary(message, message_type),
    )
    normalized_type = normalize_message_type(message, ClassificationOutput(message_type=message_type, urgency=urgency), extraction)
    extraction.message_type = normalized_type
    extraction.next_step_summary = extract_next_step_summary(message, normalized_type)
    return extraction


def normalize_message_type(
    message: str,
    classification: ClassificationOutput,
    extraction: ExtractionOutput,
) -> MessageType:
    lowered = message.lower()
    legal_terms_present = any(term in lowered for term in ("offer letter", "equity package", "terms and conditions", "agreement", "contract"))
    if extraction.compensation_mentioned or legal_terms_present:
        return MessageType.offer_related
    return classification.message_type


def decide_next_action(
    normalized_message: str,
    classification: ClassificationOutput,
    extraction: ExtractionOutput,
    candidate_profile: CandidateProfile,
    model_decision: DecisionOutput | None = None,
) -> DecisionOutput:
    missing_information = identify_missing_information(classification.message_type, extraction)
    review_reasons = identify_review_reasons(normalized_message, extraction)

    if classification.message_type == MessageType.spam_or_low_value:
        return DecisionOutput(
            recommended_action=RecommendedAction.archive_no_reply,
            needs_human_review=False,
            review_reason="",
            missing_information=[],
            explanation="The message looks low-signal or promotional, so archiving is safer than replying.",
        )

    if review_reasons:
        return DecisionOutput(
            recommended_action=RecommendedAction.escalate_human_review,
            needs_human_review=True,
            review_reason="; ".join(review_reasons),
            missing_information=missing_information,
            explanation="Sensitive terms or contradictory details make this better for manual review.",
        )

    if missing_information and classification.message_type in {
        MessageType.interview_invite,
        MessageType.scheduling_request,
        MessageType.coding_assessment,
        MessageType.recruiter_outreach,
        MessageType.networking_reply,
        MessageType.follow_up_needed,
    }:
        explanation = "A reply is warranted, but key details are missing and should be clarified first."
        if model_decision and model_decision.explanation:
            explanation = model_decision.explanation
        return DecisionOutput(
            recommended_action=RecommendedAction.ask_for_missing_info,
            needs_human_review=False,
            review_reason="",
            missing_information=missing_information,
            explanation=explanation,
        )

    explanation = "The message is actionable and does not trigger the manual review rules."
    if model_decision and model_decision.explanation:
        explanation = model_decision.explanation
    return DecisionOutput(
        recommended_action=RecommendedAction.draft_reply,
        needs_human_review=False,
        review_reason="",
        missing_information=missing_information,
        explanation=explanation,
    )


def infer_urgency_from_text(message: str) -> UrgencyLevel:
    lowered = message.lower()
    if any(token in lowered for token in ("urgent", "asap", "today", "end of day", "immediately")):
        return UrgencyLevel.critical
    if any(token in lowered for token in ("tomorrow", "deadline", "48 hours", "this afternoon")):
        return UrgencyLevel.high
    if any(token in lowered for token in ("this week", "next step", "availability", "schedule")):
        return UrgencyLevel.medium
    return UrgencyLevel.low


def extract_dates(message: str) -> list[str]:
    found: list[str] = []
    for pattern in DATE_PATTERNS:
        found.extend(re.findall(pattern, message, flags=re.IGNORECASE))
    return list(dict.fromkeys(date.strip() for date in found if date.strip()))


def identify_missing_information(message_type: MessageType, extraction: ExtractionOutput) -> list[str]:
    missing: list[str] = []
    if message_type in {MessageType.interview_invite, MessageType.scheduling_request}:
        if not extraction.dates_mentioned:
            missing.append("specific dates or time slots")
        if not extraction.timezone:
            missing.append("timezone")
        if extraction.interview_stage is None and message_type == MessageType.interview_invite:
            missing.append("interview format or stage")
    if message_type == MessageType.coding_assessment and not extraction.deadline:
        missing.append("assessment deadline")
    if message_type == MessageType.recruiter_outreach and not extraction.role_title:
        missing.append("role title")
    if message_type == MessageType.networking_reply:
        if not extraction.dates_mentioned:
            missing.append("specific time options")
        if not extraction.timezone:
            missing.append("timezone")
    return missing


def identify_review_reasons(message: str, extraction: ExtractionOutput) -> list[str]:
    reasons: list[str] = []
    if extraction.sponsorship_mentioned:
        reasons.append("visa or sponsorship language detected")
    if extraction.compensation_mentioned:
        reasons.append("compensation discussion detected")
    if "without sponsorship" in message.lower():
        reasons.append("employment eligibility language detected")
    if extraction.deadline and "?" in extraction.deadline:
        reasons.append("unclear deadline phrasing")
    if "or" in message.lower() and len(extraction.dates_mentioned) >= 2:
        reasons.append("multiple possible scheduling options need confirmation")
    if re.search(r"\bpt\b", message, flags=re.IGNORECASE) and re.search(r"\bet\b", message, flags=re.IGNORECASE):
        reasons.append("conflicting timezones detected")
    if any(term in message.lower() for term in ("terms and conditions", "agreement", "legal", "contract")):
        reasons.append("legal language detected")
    return reasons


def build_workflow_step(name: str, summary: str, status: WorkflowStatus = WorkflowStatus.completed) -> dict[str, Any]:
    return {
        "name": name,
        "status": status.value,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
    }


def simple_draft(
    normalized_message: str,
    candidate_profile: CandidateProfile,
    extraction: ExtractionOutput,
    action: RecommendedAction,
    missing_information: list[str],
) -> str:
    greeting = f"Hi {extraction.sender_name}," if extraction.sender_name else "Hi,"
    signoff = f"{candidate_profile.default_signoff},\n{candidate_profile.full_name}"
    role_phrase = f" regarding the {extraction.role_title} role" if extraction.role_title else ""
    if action == RecommendedAction.ask_for_missing_info:
        ask_line = ", ".join(missing_information) if missing_information else "a few scheduling details"
        return (
            f"{greeting}\n\n"
            f"Thank you for reaching out{role_phrase}. I would be glad to continue the conversation. "
            f"Could you please share {ask_line} so I can confirm the next step?\n\n"
            f"{signoff}"
        )
    return (
        f"{greeting}\n\n"
        f"Thank you for your message{role_phrase}. I appreciate the update and would be happy to move forward. "
        f"Please let me know the next steps and any preparation you would like me to complete.\n\n"
        f"{signoff}"
    )


def draft_usefulness_score(draft: str, extraction: ExtractionOutput) -> float:
    if not draft.strip():
        return 0.0
    score = 0.4
    lowered = draft.lower()
    if "thank" in lowered:
        score += 0.2
    if extraction.role_title and extraction.role_title.lower() in lowered:
        score += 0.2
    if extraction.sender_name and extraction.sender_name.lower().split()[0] in lowered:
        score += 0.1
    if len(draft.split()) >= 30:
        score += 0.1
    return min(score, 1.0)


def extract_company(message: str) -> str | None:
    patterns = [
        r"\bapplying to\s+([A-Z][A-Za-z0-9&.\- ]+)",
        r"\binterest in\s+([A-Z][A-Za-z0-9&.\- ]+)",
        r"\bfrom\s+([A-Z][A-Za-z0-9&.\- ]+)",
        r"\bat\s+([A-Z][A-Za-z0-9&.\- ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1).strip(" .,\n")
    return None


def extract_role_title(message: str) -> str | None:
    patterns = [
        r"\bfor the ([A-Za-z0-9/\- ,]+?) role\b",
        r"\bhiring (?:for )?an?\s+([A-Za-z0-9/\- ,]+?)(?:\s+in\b|\s+at\b|[.,\n])",
        r"\b([A-Za-z0-9/\- ,]+?) position\b",
        r"\b([A-Za-z0-9/\- ,]+?) opening\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,\n")
    return None


def extract_sender_name(message: str) -> str | None:
    signoff_match = re.search(r"(?:thanks|regards|best|sincerely),?\s*\n?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", message)
    if signoff_match:
        return signoff_match.group(1).strip()
    intro_match = re.search(r"\bI[' ]?m\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", message)
    if intro_match:
        return intro_match.group(1).strip()
    return None


def find_deadline(message: str, dates: list[str]) -> str | None:
    lowered = message.lower()
    if dates and any(token in lowered for token in ("deadline", "complete by", " by ", "before ")):
        return dates[0] if dates else "mentioned but not clearly parsed"
    if any(token in lowered for token in ("today", "tomorrow", "end of week")):
        return dates[0] if dates else "implied urgent deadline"
    return None


def extract_requested_action(message: str) -> str | None:
    lowered = message.lower()
    if "share your availability" in lowered:
        return "share availability"
    if "complete the assessment" in lowered or "take-home" in lowered:
        return "complete assessment"
    if "reply with" in lowered:
        return "reply with requested details"
    if "let me know" in lowered:
        return "respond with confirmation"
    return None


def extract_interview_stage(message: str) -> str | None:
    lowered = message.lower()
    if "phone screen" in lowered:
        return "phone_screen"
    if "technical screen" in lowered:
        return "technical_screen"
    if "onsite" in lowered:
        return "onsite"
    if "final round" in lowered:
        return "final_round"
    return None


def extract_next_step_summary(message: str, message_type: MessageType) -> str | None:
    if message_type == MessageType.rejection:
        return "No further action expected."
    if message_type == MessageType.offer_related:
        return "Review offer details before responding."
    if message_type in {MessageType.interview_invite, MessageType.scheduling_request}:
        return "Confirm the interview logistics."
    if message_type == MessageType.coding_assessment:
        return "Confirm deadline and complete the assessment."
    if message_type == MessageType.recruiter_outreach:
        return "Reply to confirm interest and gather role details."
    return None
