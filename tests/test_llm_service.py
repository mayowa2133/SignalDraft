from __future__ import annotations

from app.services.llm_service import LLMService
from app.utils.config import Settings


def test_auth_failure_disables_provider() -> None:
    service = LLMService(Settings(llm_mode="openai", openai_api_key="invalid-key", fallback_to_rules=True))
    service._handle_model_failure("Classification", Exception("Incorrect API key provided"))  # noqa: SLF001

    assert service.runtime_mode == "heuristic"
    assert service.provider_disable_reason == "OpenAI authentication failed. Falling back to heuristic mode."
