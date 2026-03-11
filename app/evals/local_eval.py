from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from app.db.database import Database
from app.db.repositories import CandidateProfileRepository, RunRepository
from app.graph.builder import SignalDraftGraph
from app.models.schemas import ExtractionOutput, RunRecord
from app.services.analysis_service import AnalysisService
from app.services.heuristics import draft_usefulness_score
from app.services.llm_service import LLMService
from app.utils.config import OUTPUT_DIR, ROOT_DIR, Settings


class ExpectedOutcome(BaseModel):
    message_type: str
    recommended_action: str
    needs_human_review: bool
    extracted: dict[str, Any] = Field(default_factory=dict)


class EvaluationExample(BaseModel):
    id: str
    title: str
    message: str
    expected: ExpectedOutcome


@dataclass(slots=True)
class LocalEvaluator:
    dataset_path: Path
    output_dir: Path

    def run(self) -> dict[str, Any]:
        settings = Settings(
            db_path=self.output_dir / "eval.db",
            checkpoint_path=self.output_dir / "eval_checkpoints.db",
        )
        settings.ensure_directories()
        database = Database(settings.db_path)
        database.initialize()
        profile_repository = CandidateProfileRepository(database)
        run_repository = RunRepository(database)
        llm_service = LLMService(settings)
        graph = SignalDraftGraph(
            llm_service=llm_service,
            profile_repository=profile_repository,
            checkpoint_path=settings.checkpoint_path,
        )
        analysis_service = AnalysisService(graph, profile_repository, run_repository)

        examples = [EvaluationExample.model_validate(item) for item in json.loads(self.dataset_path.read_text())]
        results: list[dict[str, Any]] = []
        for example in examples:
            run = analysis_service.analyze_message(example.message)
            results.append(self._score_example(example, run))

        summary = {
            "dataset_size": len(results),
            "classification_accuracy": mean(item["classification_correct"] for item in results),
            "routing_accuracy": mean(item["routing_correct"] for item in results),
            "escalation_accuracy": mean(item["escalation_correct"] for item in results),
            "extraction_completeness": mean(item["extraction_completeness"] for item in results),
            "draft_usefulness": mean(item["draft_usefulness"] for item in results),
            "examples": results,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # LangSmith dataset/evaluation hook:
        # Replace this local file write with a langsmith Client dataset upload or evaluation run
        # once you want centralized experiment tracking.

        graph.close()
        return summary

    @staticmethod
    def _score_example(example: EvaluationExample, run: RunRecord) -> dict[str, Any]:
        expected_fields = example.expected.extracted
        extraction_score = 1.0
        if expected_fields:
            field_scores: list[float] = []
            for key, expected_value in expected_fields.items():
                actual_value = run.extracted.get(key)
                if isinstance(expected_value, bool):
                    field_scores.append(1.0 if bool(actual_value) == expected_value else 0.0)
                elif isinstance(expected_value, list):
                    actual_list = [str(item).lower() for item in actual_value or []]
                    expected_list = [str(item).lower() for item in expected_value]
                    matches = sum(1 for item in expected_list if item in actual_list)
                    field_scores.append(matches / max(len(expected_list), 1))
                else:
                    actual_text = str(actual_value or "").lower()
                    expected_text = str(expected_value).lower()
                    field_scores.append(1.0 if expected_text in actual_text else 0.0)
            extraction_score = mean(field_scores)

        if example.expected.recommended_action in {"draft_reply", "ask_for_missing_info"}:
            draft_score = draft_usefulness_score(
                run.draft_reply,
                ExtractionOutput.model_validate(run.extracted),
            )
        else:
            draft_score = 1.0 if not run.draft_reply.strip() or run.needs_human_review else 0.8

        return {
            "id": example.id,
            "title": example.title,
            "predicted_message_type": run.message_type.value,
            "predicted_action": run.recommended_action.value,
            "predicted_review": run.needs_human_review,
            "classification_correct": 1.0 if run.message_type.value == example.expected.message_type else 0.0,
            "routing_correct": 1.0 if run.recommended_action.value == example.expected.recommended_action else 0.0,
            "escalation_correct": 1.0 if run.needs_human_review == example.expected.needs_human_review else 0.0,
            "extraction_completeness": extraction_score,
            "draft_usefulness": draft_score,
            "review_reason": run.review_reason,
        }


def run_local_eval() -> dict[str, Any]:
    dataset_path = ROOT_DIR / "data" / "eval_dataset.json"
    output_dir = OUTPUT_DIR / "evals"
    evaluator = LocalEvaluator(dataset_path=dataset_path, output_dir=output_dir)
    return evaluator.run()


if __name__ == "__main__":
    print(json.dumps(run_local_eval(), indent=2))
