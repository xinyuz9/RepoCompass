"""Isolated subprocess fixture: real graph/SQLite, deterministic models, forced exit."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.memory.store import open_memory_store
from repocompass.service import (
    InvestigationService,
    StartInvestigationRequest,
    open_service_operation_store,
)
from repocompass.storage.cancellation import open_cancellation_store
from repocompass.storage.raw_store import open_raw_run_store
from tests.model_doubles import with_scripted_planner


def main():
    root = Path(sys.argv[1])
    terminal = sys.argv[2] == "terminal"
    suffix = "" if len(sys.argv) < 4 else "-" + sys.argv[3]
    explorer = MagicMock(spec=BaseChatModel)
    responses = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_code",
                        "args": {
                            "relative_path": "config.py",
                            "start_line": 1,
                            "end_line": 1,
                        },
                        "id": "read-flag",
                    }
                ],
            ),
            AIMessage(content="Enough source evidence."),
        ]
    )
    explorer.bind_tools.return_value = RunnableLambda(lambda _: next(responses))
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: CandidateResult(
            evidence=[
                EvidenceItem(
                    path="config.py", line_start=1, line_end=1, claim="FLAG is True."
                )
            ],
            summary="FLAG is True.",
            unresolved_questions=[],
        )
    )
    models = with_scripted_planner(explorer)
    if terminal:
        import repocompass.memory.consolidation as consolidation

        consolidation.consolidate_terminal_episode = lambda *args, **kwargs: os._exit(
            19
        )
    else:
        models.planner.with_structured_output.return_value = RunnableLambda(
            lambda _: os._exit(19)
        )
    with (
        open_sqlite_checkpointer(root / "checkpoints.sqlite") as checkpoint,
        open_raw_run_store(root / "raw.sqlite") as raw,
        open_memory_store(root / "memory.sqlite") as memory,
        open_service_operation_store(root / "operations.sqlite") as operations,
        open_cancellation_store(root / "cancellations.sqlite") as cancellation,
    ):
        runner = AgentRunner(
            models,
            checkpointer=checkpoint,
            raw_run_store=raw,
            memory_store=memory,
            cancellation_store=cancellation,
            run_id_factory=lambda: "orphan-run" + suffix,
        )
        service = InvestigationService(
            runner,
            operation_store=operations,
            raw_run_store=raw,
            thread_id_factory=lambda: "orphan-thread" + suffix,
        )
        service.start_investigation(
            StartInvestigationRequest(
                request_id="orphan-request" + suffix,
                repo_path=root / "repo",
                objective="Explain the flag.",
                memory_retrieval_enabled=True,
                context_budget_tokens=16384,
            )
        )
    raise AssertionError("worker was expected to exit before ledger completion")


if __name__ == "__main__":
    main()
