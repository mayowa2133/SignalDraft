from __future__ import annotations

from app.db.database import Database
from app.db.repositories import CandidateProfileRepository, RunRepository
from app.graph.builder import SignalDraftGraph
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMService
from app.models.schemas import ReadinessResponse
from app.utils.config import settings
from app.utils.logging import configure_logging


class AppContainer:
    def __init__(self) -> None:
        configure_logging()
        settings.ensure_directories()
        self.database = Database(settings.db_path)
        self.database.initialize()
        self.profile_repository = CandidateProfileRepository(self.database)
        self.run_repository = RunRepository(self.database)
        self.llm_service = LLMService(settings)
        self.graph = SignalDraftGraph(
            llm_service=self.llm_service,
            profile_repository=self.profile_repository,
            checkpoint_path=settings.checkpoint_path,
        )
        self.analysis_service = AnalysisService(
            graph=self.graph,
            llm_service=self.llm_service,
            profile_repository=self.profile_repository,
            run_repository=self.run_repository,
        )

    def close(self) -> None:
        self.graph.close()

    def readiness_status(self) -> ReadinessResponse:
        db_writable = self.database.is_writable()
        provider_ready = settings.llm_mode != "openai" or self.llm_service.runtime_mode == "openai"
        status = "ready" if db_writable and provider_ready else "degraded"
        return ReadinessResponse(
            status=status,
            environment=settings.environment,
            llm_mode_requested=settings.llm_mode,
            llm_runtime_mode=self.llm_service.runtime_mode,
            fallback_to_rules=settings.fallback_to_rules,
            openai_key_present=bool(settings.openai_api_key),
            backend_auth_enabled=settings.backend_auth_enabled,
            db_writable=db_writable,
            provider_disable_reason=self.llm_service.provider_disable_reason,
        )


container = AppContainer()
