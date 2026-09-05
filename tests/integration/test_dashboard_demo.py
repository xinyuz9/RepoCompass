"""End-to-end sanitized dashboard demo generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.demo import (
    DEMO_RUN_ID,
    DEMO_THREAD_ID,
    DashboardDemoError,
    create_dashboard_demo,
)
from repocompass.dashboard.runtime import create_inspection_models
from repocompass.dashboard.service import (
    DashboardQueryService,
    DashboardSnapshotConflictError,
)
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.contracts import MemoryScope
from repocompass.memory.store import open_memory_store
from repocompass.storage.raw_store import open_raw_run_store
from repocompass.structure import StructuralVerificationStatus


def test_real_sqlite_continuation_exposes_public_history_after_restart(tmp_path):
    workspace = create_dashboard_demo(tmp_path / "multi", turns=3)
    with (
        open_sqlite_checkpointer(workspace.checkpoint_database) as checkpointer,
        open_raw_run_store(workspace.raw_store_database) as raw_store,
    ):
        runner = AgentRunner(create_inspection_models(), checkpointer=checkpointer)
        service = DashboardQueryService(
            runner,
            raw_run_store=raw_store,
            trace_store=JsonlTraceStore(workspace.trace_directory),
            repository_root=workspace.repository_path,
        )
        recent = service.get_conversation(DEMO_THREAD_ID, before_turn=3, limit=1)
        older = service.get_conversation(
            DEMO_THREAD_ID, before_turn=recent.next_before_turn, limit=1
        )
        assert [turn.turn_number for turn in recent.turns] == [2]
        assert [turn.turn_number for turn in older.turns] == [1]
        assert older.next_before_turn is None
        assert all(turn.summary for turn in (*older.turns, *recent.turns))
        assert str(workspace.repository_path) not in recent.model_dump_json()
        assert (
            len(raw_store.list_attempts(thread_id=DEMO_THREAD_ID, max_results=10)) == 3
        )


def test_demo_uses_production_stores_and_populates_every_observability_lens(
    tmp_path: Path,
) -> None:
    workspace = create_dashboard_demo(tmp_path / "demo")

    with (
        open_sqlite_checkpointer(workspace.checkpoint_database) as checkpointer,
        open_raw_run_store(workspace.raw_store_database) as raw_store,
        open_memory_store(workspace.memory_database) as memory_store,
    ):
        runner = AgentRunner(
            create_inspection_models(),
            checkpointer=checkpointer,
            raw_run_store=raw_store,
            memory_store=memory_store,
            pricing_table=workspace.pricing_table,
        )
        service = DashboardQueryService(
            runner,
            raw_run_store=raw_store,
            trace_store=JsonlTraceStore(workspace.trace_directory),
            repository_root=workspace.repository_path,
            memory_store=memory_store,
            pricing_table=workspace.pricing_table,
            demo_mode=True,
        )

        system = service.system_info()
        catalog = service.list_threads()
        thread = service.get_thread(DEMO_THREAD_ID)
        run = service.get_run(DEMO_RUN_ID)
        memory = service.list_memory()
        policy_step = next(
            step
            for step in thread.thread.structure.reading_path.steps
            if step.path == "pkg/policy.py"
        )
        source = service.get_source(DEMO_THREAD_ID, policy_step.entity_id)

    assert system.demo_mode is True
    assert system.lifecycle_mutations_enabled is False
    assert system.memory_mutations_enabled is False
    assert [item.thread_id for item in catalog.threads] == [DEMO_THREAD_ID]
    assert thread.thread.execution_state.value == "completed"
    assert thread.thread.plan is not None
    assert len(thread.thread.plan.steps) == 3
    assert thread.thread.structure is not None
    assert thread.thread.structure.indexed_paths == (
        "pkg/api.py",
        "pkg/policy.py",
        "tests/test_api.py",
    )
    verified_edges = [
        edge
        for edge in thread.thread.structure.reading_path.graph_slice.edges
        if edge.verification_status is StructuralVerificationStatus.VERIFIED
    ]
    assert verified_edges
    assert len(run.trace_events) >= 30
    assert len(run.plan_timeline) >= 3
    assert len(run.evaluator_timeline) == 3
    assert len(run.context_packs) == len(run.model_invocations) == 15
    assert thread.thread.final_result is not None
    assert not hasattr(thread.thread.final_result, "structure")
    assert thread.thread.structure is not None
    assert not hasattr(thread.thread.structure, "graph")
    assert "context_pack" not in thread.model_dump_json()
    assert len(thread.model_dump_json()) < 175_000
    assert all(item.provider == "demo" for item in run.model_invocations)
    assert run.usage_cost_report is not None
    assert run.usage_cost_report.breakdowns[0].cost.estimated_cost_usd != "0"
    assert any(
        item.source.memory is not None
        for pack in run.context_packs
        for item in pack.items
    )
    assert {item.memory_scope for item in memory.entries} == {
        MemoryScope.REPOSITORY,
        MemoryScope.EPISODIC,
    }
    assert str(workspace.root) not in run.model_dump_json()
    assert source.path == "pkg/policy.py"
    assert source.source_digest == policy_step.source_locations[0].source_digest
    assert source.repository_id == system.repository_id
    assert any("class ProposalVerifier" in line.text for line in source.lines)
    assert source.evidence_line_start <= source.evidence_line_end


def test_demo_source_preview_fails_closed_after_content_drift(tmp_path: Path) -> None:
    workspace = create_dashboard_demo(tmp_path / "demo")

    with (
        open_sqlite_checkpointer(workspace.checkpoint_database) as checkpointer,
        open_raw_run_store(workspace.raw_store_database) as raw_store,
        open_memory_store(workspace.memory_database) as memory_store,
    ):
        service = DashboardQueryService(
            AgentRunner(
                create_inspection_models(),
                checkpointer=checkpointer,
                raw_run_store=raw_store,
                memory_store=memory_store,
                pricing_table=workspace.pricing_table,
            ),
            raw_run_store=raw_store,
            trace_store=JsonlTraceStore(workspace.trace_directory),
            repository_root=workspace.repository_path,
            memory_store=memory_store,
            pricing_table=workspace.pricing_table,
        )
        thread = service.get_thread(DEMO_THREAD_ID)
        policy_step = next(
            step
            for step in thread.thread.structure.reading_path.steps
            if step.path == "pkg/policy.py"
        )
        policy_path = workspace.repository_path / "pkg" / "policy.py"
        policy_path.write_text(
            policy_path.read_text(encoding="utf-8") + "\n# drift\n",
            encoding="utf-8",
        )

        with pytest.raises(DashboardSnapshotConflictError, match="content changed"):
            service.get_source(DEMO_THREAD_ID, policy_step.entity_id)


def test_demo_refuses_to_overlay_any_existing_state(tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("user-owned", encoding="utf-8")

    with pytest.raises(DashboardDemoError, match="empty"):
        create_dashboard_demo(destination)

    assert marker.read_text(encoding="utf-8") == "user-owned"
