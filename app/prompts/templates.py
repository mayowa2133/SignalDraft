from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You classify inbound career-related messages for a job-seeker assistant. "
                "Choose the single best category, estimate urgency conservatively, and explain briefly. "
                "Do not infer facts that are not present."
            ),
        ),
        (
            "human",
            (
                "Classify the message below.\n\n"
                "Allowed categories: recruiter_outreach, interview_invite, scheduling_request, "
                "coding_assessment, rejection, offer_related, networking_reply, follow_up_needed, "
                "spam_or_low_value, unknown.\n\n"
                "Message:\n{message}"
            ),
        ),
    ]
)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Extract structured recruiting and application details from the message. "
                "Return only facts grounded in the message. Use null when a field is missing."
            ),
        ),
        (
            "human",
            (
                "Message type hint: {message_type}\n"
                "Urgency hint: {urgency}\n\n"
                "Message:\n{message}"
            ),
        ),
    ]
)


DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You support an inbox-triage agent for job seekers. "
                "Recommend the safest next action. Escalate only when there is real risk or ambiguity. "
                "Flag missing details when a reply should request them."
            ),
        ),
        (
            "human",
            (
                "Raw message:\n{message}\n\n"
                "Structured extraction:\n{extracted}\n\n"
                "Available actions: draft_reply, ask_for_missing_info, archive_no_reply, escalate_human_review.\n"
                "Return the best action, whether a human review is needed, what information is missing, "
                "and a concise explanation."
            ),
        ),
    ]
)


DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Write a concise, professional, human-sounding email reply for a job seeker. "
                "Do not fabricate specifics, commitments, compensation expectations, or sponsorship terms. "
                "Match the user's preferred tone while staying safe and realistic."
            ),
        ),
        (
            "human",
            (
                "Candidate profile:\n{candidate_profile}\n\n"
                "Message type: {message_type}\n"
                "Recommended action: {recommended_action}\n"
                "Missing information: {missing_information}\n"
                "Structured extraction:\n{extracted}\n\n"
                "Original message:\n{message}\n\n"
                "Draft a response that is ready for manual approval. "
                "If important details are missing, ask clear follow-up questions."
            ),
        ),
    ]
)


SAFETY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Review the drafted reply for risk, overclaiming, hallucinated details, and tone issues. "
                "Escalate if the message involves compensation negotiation, sponsorship commitments, "
                "legal risk, contradictory scheduling details, or unclear deadlines."
            ),
        ),
        (
            "human",
            (
                "Original message:\n{message}\n\n"
                "Structured extraction:\n{extracted}\n\n"
                "Draft reply:\n{draft_reply}\n\n"
                "Review the draft and revise it only if necessary."
            ),
        ),
    ]
)
