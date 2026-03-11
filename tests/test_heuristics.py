from __future__ import annotations

from app.models.schemas import CandidateProfile, MessageType
from app.services.heuristics import (
    classify_with_rules,
    decide_next_action,
    extract_with_rules,
    normalize_message,
    normalize_message_type,
)


def test_normalize_message_collapses_whitespace() -> None:
    raw = "Hi Alex,\n\n  Please   share   your availability.\n"
    assert normalize_message(raw) == "Hi Alex, Please share your availability."


def test_sensitive_message_escalates_to_human_review() -> None:
    message = (
        "Hi Alex, can you confirm if you need visa sponsorship and share your "
        "salary expectations for a base salary of $180,000?"
    )
    classification = classify_with_rules(message)
    extraction = extract_with_rules(message, classification)
    decision = decide_next_action(
        normalized_message=message,
        classification=classification,
        extraction=extraction,
        candidate_profile=CandidateProfile(),
    )
    assert decision.recommended_action.value == "escalate_human_review"
    assert decision.needs_human_review is True


def test_missing_scheduling_details_asks_for_more_info() -> None:
    message = "We would like to schedule your interview next week. Please send availability."
    classification = classify_with_rules(message)
    extraction = extract_with_rules(message, classification)
    decision = decide_next_action(
        normalized_message=message,
        classification=classification,
        extraction=extraction,
        candidate_profile=CandidateProfile(),
    )
    assert decision.recommended_action.value == "ask_for_missing_info"
    assert "timezone" in decision.missing_information


def test_spam_is_archived() -> None:
    message = "Book a demo for our outbound package and unsubscribe anytime."
    classification = classify_with_rules(message)
    extraction = extract_with_rules(message, classification)
    decision = decide_next_action(
        normalized_message=message,
        classification=classification,
        extraction=extraction,
        candidate_profile=CandidateProfile(),
    )
    assert decision.recommended_action.value == "archive_no_reply"


def test_compensation_screen_is_normalized_to_offer_related() -> None:
    message = (
        "Hi Alex, we are excited about your profile for the Senior ML Engineer opportunity at Atlas Commerce. "
        "Before we move ahead, can you confirm whether you require visa sponsorship and share your compensation "
        "expectations for a base salary range of $170,000-$190,000?"
    )
    model_like_classification = classify_with_rules("We have an opportunity and would love to talk.")
    model_like_classification.message_type = MessageType.recruiter_outreach
    extraction = extract_with_rules(message, model_like_classification)
    normalized_type = normalize_message_type(message, model_like_classification, extraction)
    assert normalized_type.value == "offer_related"
