from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

load_dotenv(ROOT_DIR / ".env")


def _parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.lower() == "true"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("SIGNALDRAFT_APP_NAME", "SignalDraft")
    environment: str = os.getenv("SIGNALDRAFT_ENV", "local")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    llm_mode: str = os.getenv("SIGNALDRAFT_LLM_MODE", "openai")
    fallback_to_rules: bool = _parse_bool(os.getenv("SIGNALDRAFT_FALLBACK_TO_RULES", "true"), default=True)
    db_path: Path = Path(os.getenv("SIGNALDRAFT_DB_PATH", str(DATA_DIR / "signaldraft.db")))
    checkpoint_path: Path = Path(
        os.getenv("SIGNALDRAFT_CHECKPOINT_PATH", str(DATA_DIR / "signaldraft_checkpoints.db"))
    )
    api_base_url: str = os.getenv("SIGNALDRAFT_API_BASE_URL", "http://127.0.0.1:8000")
    public_ui_url: str = os.getenv("SIGNALDRAFT_PUBLIC_UI_URL", "http://127.0.0.1:8501")
    allowed_origins_raw: str = os.getenv(
        "SIGNALDRAFT_ALLOWED_ORIGINS",
        "http://127.0.0.1:8501,http://localhost:8501",
    )
    api_token: str = os.getenv("SIGNALDRAFT_API_TOKEN", "")
    admin_password: str = os.getenv("SIGNALDRAFT_ADMIN_PASSWORD", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "SignalDraft")

    def ensure_directories(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def allowed_origins(self) -> list[str]:
        origins = _split_csv(self.allowed_origins_raw)
        if self.public_ui_url and self.public_ui_url not in origins:
            origins.append(self.public_ui_url)
        return origins

    @property
    def resolved_api_base_url(self) -> str:
        if "://" in self.api_base_url:
            return self.api_base_url.rstrip("/")
        return f"http://{self.api_base_url.rstrip('/')}"

    @property
    def backend_auth_enabled(self) -> bool:
        return bool(self.api_token)

    @property
    def admin_auth_enabled(self) -> bool:
        return bool(self.admin_password)


settings = Settings()
settings.ensure_directories()
