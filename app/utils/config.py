from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

load_dotenv(ROOT_DIR / ".env")


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("SIGNALDRAFT_APP_NAME", "SignalDraft")
    environment: str = os.getenv("SIGNALDRAFT_ENV", "local")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    llm_mode: str = os.getenv("SIGNALDRAFT_LLM_MODE", "openai")
    fallback_to_rules: bool = os.getenv("SIGNALDRAFT_FALLBACK_TO_RULES", "true").lower() == "true"
    db_path: Path = Path(os.getenv("SIGNALDRAFT_DB_PATH", str(DATA_DIR / "signaldraft.db")))
    checkpoint_path: Path = Path(
        os.getenv("SIGNALDRAFT_CHECKPOINT_PATH", str(DATA_DIR / "signaldraft_checkpoints.db"))
    )
    api_base_url: str = os.getenv("SIGNALDRAFT_API_BASE_URL", "http://127.0.0.1:8000")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "SignalDraft")

    def ensure_directories(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

