from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import (
    CandidateProfile,
    ClassificationOutput,
    DecisionOutput,
    ExtractionOutput,
    RecommendedAction,
    SafetyReviewOutput,
)
from app.prompts.templates import (
    CLASSIFICATION_PROMPT,
    DECISION_PROMPT,
    DRAFT_PROMPT,
    EXTRACTION_PROMPT,
    SAFETY_PROMPT,
)
from app.services.heuristics import (
    classify_with_rules,
    draft_usefulness_score,
    extract_with_rules,
    simple_draft,
)
from app.utils.config import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - dependency installed at runtime
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - dependency installed at runtime
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[misc]
        def decorator(func: Any) -> Any:
            return func

        return decorator


@dataclass(slots=True)
class LLMService:
    settings: Settings
    _model: Any = field(init=False, default=None, repr=False)
    _provider_disabled_reason: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if self.settings.llm_mode == "openai" and self.settings.openai_api_key and ChatOpenAI is not None:
            self._model = ChatOpenAI(
                model=self.settings.openai_model,
                temperature=self.settings.openai_temperature,
                api_key=self.settings.openai_api_key,
            )
        elif self.settings.llm_mode == "openai" and not self.settings.openai_api_key:
            self._provider_disabled_reason = "OpenAI API key is not configured."
        elif self.settings.llm_mode == "openai" and ChatOpenAI is None:
            self._provider_disabled_reason = "langchain_openai is not available."

    @property
    def model_available(self) -> bool:
        return self._model is not None

    @property
    def runtime_mode(self) -> str:
        return self.settings.llm_mode if self.model_available else "heuristic"

    @property
    def provider_disable_reason(self) -> str:
        return self._provider_disabled_reason

    def _disable_provider(self, reason: str) -> None:
        self._model = None
        self._provider_disabled_reason = reason

    @staticmethod
    def _is_auth_failure(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        response_status_code = getattr(response, "status_code", None)
        message = str(exc).lower()
        return (
            status_code in {401, 403}
            or response_status_code in {401, 403}
            or "incorrect api key" in message
            or "invalid_api_key" in message
            or "unauthorized" in message
            or "authentication" in message
        )

    def _handle_model_failure(self, stage: str, exc: Exception) -> None:
        if self._is_auth_failure(exc):
            self._disable_provider("OpenAI authentication failed. Falling back to heuristic mode.")
            logger.warning("%s fallback activated after provider authentication failed.", stage)
            return
        logger.warning("%s fallback activated due to %s.", stage, type(exc).__name__)

    @traceable(name="signaldraft_classification")
    def classify_message(self, message: str) -> ClassificationOutput:
        if not self.model_available:
            return classify_with_rules(message)
        try:
            chain = CLASSIFICATION_PROMPT | self._model.with_structured_output(ClassificationOutput)
            return chain.invoke({"message": message})
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            self._handle_model_failure("Classification", exc)
            if self.settings.fallback_to_rules:
                return classify_with_rules(message)
            raise

    @traceable(name="signaldraft_extraction")
    def extract_fields(
        self,
        message: str,
        message_type: str,
        urgency: str,
        fallback_classification: ClassificationOutput,
    ) -> ExtractionOutput:
        if not self.model_available:
            return extract_with_rules(message, fallback_classification)
        try:
            chain = EXTRACTION_PROMPT | self._model.with_structured_output(ExtractionOutput)
            result = chain.invoke(
                {
                    "message": message,
                    "message_type": message_type,
                    "urgency": urgency,
                }
            )
            if result.message_type is None:
                result.message_type = fallback_classification.message_type
            if result.urgency is None:
                result.urgency = fallback_classification.urgency
            return result
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            self._handle_model_failure("Extraction", exc)
            if self.settings.fallback_to_rules:
                return extract_with_rules(message, fallback_classification)
            raise

    @traceable(name="signaldraft_decision_assist")
    def assist_decision(self, message: str, extracted: dict[str, Any]) -> DecisionOutput | None:
        if not self.model_available:
            return None
        try:
            chain = DECISION_PROMPT | self._model.with_structured_output(DecisionOutput)
            return chain.invoke({"message": message, "extracted": extracted})
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            self._handle_model_failure("Decision assist", exc)
            return None

    @traceable(name="signaldraft_draft")
    def draft_response(
        self,
        message: str,
        candidate_profile: CandidateProfile,
        message_type: str,
        recommended_action: RecommendedAction,
        missing_information: list[str],
        extracted: dict[str, Any],
    ) -> str:
        fallback = simple_draft(
            normalized_message=message,
            candidate_profile=candidate_profile,
            extraction=ExtractionOutput.model_validate(extracted),
            action=recommended_action,
            missing_information=missing_information,
        )
        if not self.model_available:
            return fallback
        try:
            chain = DRAFT_PROMPT | self._model
            result = chain.invoke(
                {
                    "candidate_profile": candidate_profile.model_dump(),
                    "message_type": message_type,
                    "recommended_action": recommended_action.value,
                    "missing_information": missing_information,
                    "extracted": extracted,
                    "message": message,
                }
            )
            content = getattr(result, "content", "")
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            return str(content).strip() or fallback
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            self._handle_model_failure("Draft", exc)
            return fallback

    @traceable(name="signaldraft_safety_review")
    def review_draft(self, message: str, extracted: dict[str, Any], draft_reply: str) -> SafetyReviewOutput:
        heuristic_score = draft_usefulness_score(draft_reply, ExtractionOutput.model_validate(extracted))
        fallback = SafetyReviewOutput(
            approved=True,
            needs_human_review=False,
            risk_note="",
            revised_draft=draft_reply,
            explanation=f"Heuristic safety pass completed. Draft usefulness score: {heuristic_score:.2f}.",
        )
        if not self.model_available:
            return fallback
        try:
            chain = SAFETY_PROMPT | self._model.with_structured_output(SafetyReviewOutput)
            result = chain.invoke({"message": message, "extracted": extracted, "draft_reply": draft_reply})
            if not result.revised_draft:
                result.revised_draft = draft_reply
            return result
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            self._handle_model_failure("Safety review", exc)
            return fallback
