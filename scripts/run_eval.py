from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evals.local_eval import run_local_eval


if __name__ == "__main__":
    print(json.dumps(run_local_eval(), indent=2))
