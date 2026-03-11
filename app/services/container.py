from __future__ import annotations

from app.db.database import Database
from app.db.repositories import CandidateProfileRepository, RunRepository
from app.graph.builder import SignalDraftGraph
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMService
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
            profile_repository=self.profile_repository,
            run_repository=self.run_repository,
        )

    def close(self) -> None:
        self.graph.close()


container = AppContainer()

