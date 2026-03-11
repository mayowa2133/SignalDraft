from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.db.repositories import CandidateProfileRepository
from app.graph.nodes import (
    finalize_result_node,
    ingest_message_node,
    make_classify_message_node,
    make_decide_action_node,
    make_draft_response_node,
    make_extract_fields_node,
    make_load_candidate_context_node,
    make_safety_review_node,
)
from app.graph.state import AgentState
from app.models.schemas import RecommendedAction
from app.services.llm_service import LLMService


class GraphCheckpointer:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self._manager: AbstractContextManager[Any] | None = None
        self._checkpointer: Any = None

    @property
    def saver(self) -> Any:
        if self._checkpointer is None:
            self._checkpointer = self._build()
        return self._checkpointer

    def close(self) -> None:
        if self._manager is not None:
            self._manager.__exit__(None, None, None)
            self._manager = None

    def _build(self) -> Any:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            self._manager = SqliteSaver.from_conn_string(str(self.checkpoint_path))
            return self._manager.__enter__()
        except Exception:
            try:
                from langgraph.checkpoint.memory import InMemorySaver

                return InMemorySaver()
            except ImportError:
                from langgraph.checkpoint.memory import MemorySaver

                return MemorySaver()


class SignalDraftGraph:
    def __init__(
        self,
        llm_service: LLMService,
        profile_repository: CandidateProfileRepository,
        checkpoint_path: Path,
    ) -> None:
        self.checkpointer = GraphCheckpointer(checkpoint_path)
        self.graph = self._compile_graph(llm_service, profile_repository)

    def invoke(self, state: AgentState, run_id: str) -> AgentState:
        return self.graph.invoke(state, config={"configurable": {"thread_id": run_id}})

    def close(self) -> None:
        self.checkpointer.close()

    def _compile_graph(
        self,
        llm_service: LLMService,
        profile_repository: CandidateProfileRepository,
    ) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("ingest_message", ingest_message_node)
        workflow.add_node("classify_message", make_classify_message_node(llm_service))
        workflow.add_node("extract_fields", make_extract_fields_node(llm_service))
        workflow.add_node("load_candidate_context", make_load_candidate_context_node(profile_repository))
        workflow.add_node("decide_action", make_decide_action_node(llm_service))
        workflow.add_node("draft_response", make_draft_response_node(llm_service))
        workflow.add_node("safety_review", make_safety_review_node(llm_service))
        workflow.add_node("finalize_result", finalize_result_node)

        workflow.add_edge(START, "ingest_message")
        workflow.add_edge("ingest_message", "classify_message")
        workflow.add_edge("classify_message", "extract_fields")
        workflow.add_edge("extract_fields", "load_candidate_context")
        workflow.add_edge("load_candidate_context", "decide_action")
        workflow.add_conditional_edges(
            "decide_action",
            self._route_after_decision,
            {
                "draft_path": "draft_response",
                "finalize_path": "finalize_result",
            },
        )
        workflow.add_edge("draft_response", "safety_review")
        workflow.add_edge("safety_review", "finalize_result")
        workflow.add_edge("finalize_result", END)

        return workflow.compile(checkpointer=self.checkpointer.saver)

    @staticmethod
    def _route_after_decision(state: AgentState) -> str:
        action = state.get("recommended_action", RecommendedAction.draft_reply.value)
        if action in {
            RecommendedAction.draft_reply.value,
            RecommendedAction.ask_for_missing_info.value,
        }:
            return "draft_path"
        return "finalize_path"

