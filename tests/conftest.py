from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_OUTPUT_DIR = ROOT_DIR / "outputs" / "test_runtime"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("SIGNALDRAFT_LLM_MODE", "heuristic")
os.environ.setdefault("SIGNALDRAFT_FALLBACK_TO_RULES", "true")
os.environ.setdefault("SIGNALDRAFT_API_TOKEN", "test-api-token")
os.environ.setdefault("SIGNALDRAFT_ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("SIGNALDRAFT_DB_PATH", str(TEST_OUTPUT_DIR / "signaldraft-test.db"))
os.environ.setdefault(
    "SIGNALDRAFT_CHECKPOINT_PATH",
    str(TEST_OUTPUT_DIR / "signaldraft-checkpoints.db"),
)
