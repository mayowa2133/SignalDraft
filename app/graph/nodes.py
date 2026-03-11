from __future__ import annotations

from typing import Callable

from app.db.repositories import CandidateProfileRepository
from app.graph.state import AgentState
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
from app.services.heuristics import (
    build_workflow_step,
    decide_next_action,
    extract_next_step_summary,
    normalize_message,
    normalize_message_type,
)
from app.services.llm_service import LLMService


def _append_step(state: AgentState, name: str, summary: str, status: WorkflowStatus = WorkflowStatus.completed) -> list[dict]:
    existing = list(state.get("workflow_steps", []))
    existing.append(build_workflow_step(name=name, summary=summary, status=status))
    return existing


def ingest_message_node(state: AgentState) -> AgentState:
    normalized = normalize_message(state["raw_message"])
    return {
        "normalized_message": normalized,
        "errors": list(state.get("errors", [])),
        "workflow_steps": _append_step(state, "ingest_message", "Normalized whitespace and prepared the message."),
    }


def make_classify_message_node(llm_service: LLMService) -> Callable[[AgentState], AgentState]:
    def classify_message_node(state: AgentState) -> AgentState:
        classification = llm_service.classify_message(state["normalized_message"])
        summary = f"Classified as {classification.message_type.value} with {classification.urgency.value} urgency."
        return {
            "message_type": classification.message_type.value,
            "urgency": classification.urgency.value,
            "workflow_steps": _append_step(state, "classify_message", summary),
        }

    return classify_message_node


def make_extract_fields_node(llm_service: LLMService) -> Callable[[AgentState], AgentState]:
    def extract_fields_node(state: AgentState) -> AgentState:
        fallback_classification = ClassificationOutput(
            message_type=MessageType(state["message_type"]),
            urgency=UrgencyLevel(state["urgency"]),
        )
        extraction = llm_service.extract_fields(
            message=state["normalized_message"],
            message_type=state["message_type"],
            urgency=state["urgency"],
            fallback_classification=fallback_classification,
        )
        normalized_type = normalize_message_type(state["normalized_message"], fallback_classification, extraction)
        summary = "Extracted structured fields for routing and drafting."
        if normalized_type != fallback_classification.message_type:
            extraction.message_type = normalized_type
            extraction.next_step_summary = extract_next_step_summary(state["normalized_message"], normalized_type)
            summary = f"Extracted structured fields and normalized message type to {normalized_type.value}."
        return {
            "message_type": normalized_type.value,
            "extracted": extraction.model_dump(mode="json"),
            "workflow_steps": _append_step(state, "extract_fields", summary),
        }

    return extract_fields_node


def make_load_candidate_context_node(profile_repository: CandidateProfileRepository) -> Callable[[AgentState], AgentState]:
    def load_candidate_context_node(state: AgentState) -> AgentState:
        profile = profile_repository.get()
        summary = f"Loaded candidate profile for {profile.full_name}."
        return {
            "candidate_profile": profile.model_dump(mode="json"),
            "workflow_steps": _append_step(state, "load_candidate_context", summary),
        }

    return load_candidate_context_node


def make_decide_action_node(llm_service: LLMService) -> Callable[[AgentState], AgentState]:
    def decide_action_node(state: AgentState) -> AgentState:
        model_decision = llm_service.assist_decision(state["normalized_message"], state["extracted"])
        candidate_profile = CandidateProfile.model_validate(state["candidate_profile"])
        classification = ClassificationOutput(
            message_type=MessageType(state["message_type"]),
            urgency=UrgencyLevel(state["urgency"]),
        )
        extraction = ExtractionOutput.model_validate(state["extracted"])
        final_decision: DecisionOutput = decide_next_action(
            normalized_message=state["normalized_message"],
            classification=classification,
            extraction=extraction,
            candidate_profile=candidate_profile,
            model_decision=model_decision,
        )
        summary = f"Selected {final_decision.recommended_action.value}."
        if final_decision.needs_human_review and final_decision.review_reason:
            summary = f"{summary} Review reason: {final_decision.review_reason}."
        return {
            "recommended_action": final_decision.recommended_action.value,
            "needs_human_review": final_decision.needs_human_review,
            "review_reason": final_decision.review_reason,
            "explanation": final_decision.explanation,
            "missing_information": final_decision.missing_information,
            "workflow_steps": _append_step(state, "decide_action", summary),
        }

    return decide_action_node


def make_draft_response_node(llm_service: LLMService) -> Callable[[AgentState], AgentState]:
    def draft_response_node(state: AgentState) -> AgentState:
        candidate_profile = CandidateProfile.model_validate(state["candidate_profile"])
        action = RecommendedAction(state["recommended_action"])
        draft = llm_service.draft_response(
            message=state["normalized_message"],
            candidate_profile=candidate_profile,
            message_type=state["message_type"],
            recommended_action=action,
            missing_information=list(state.get("missing_information", [])),
            extracted=state["extracted"],
        )
        summary = "Generated a draft response for manual review."
        return {
            "draft_reply": draft,
            "workflow_steps": _append_step(state, "draft_response", summary),
        }

    return draft_response_node


def make_safety_review_node(llm_service: LLMService) -> Callable[[AgentState], AgentState]:
    def safety_review_node(state: AgentState) -> AgentState:
        review = llm_service.review_draft(
            message=state["normalized_message"],
            extracted=state["extracted"],
            draft_reply=state.get("draft_reply", ""),
        )
        recommended_action = state["recommended_action"]
        review_reason = state.get("review_reason", "")
        explanation = state.get("explanation", "")
        if review.needs_human_review:
            recommended_action = RecommendedAction.escalate_human_review.value
            review_reason = review.risk_note or review_reason
        if review.explanation:
            explanation = review.explanation
        summary = "Completed final safety review."
        status = WorkflowStatus.warning if review.needs_human_review else WorkflowStatus.completed
        return {
            "draft_reply": review.revised_draft or state.get("draft_reply", ""),
            "recommended_action": recommended_action,
            "needs_human_review": review.needs_human_review or state.get("needs_human_review", False),
            "review_reason": review_reason,
            "explanation": explanation,
            "workflow_steps": _append_step(state, "safety_review", summary, status=status),
        }

    return safety_review_node


def finalize_result_node(state: AgentState) -> AgentState:
    action = state.get("recommended_action", RecommendedAction.draft_reply.value)
    summary = f"Packaged final result for {action}."
    if action in {RecommendedAction.archive_no_reply.value, RecommendedAction.escalate_human_review.value}:
        draft_reply = state.get("draft_reply", "")
    else:
        draft_reply = state.get("draft_reply", "")
    return {
        "draft_reply": draft_reply,
        "workflow_steps": _append_step(state, "finalize_result", summary),
    }
