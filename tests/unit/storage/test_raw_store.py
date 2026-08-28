"""Transactional Raw Run Store behavior and provenance integration tests."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.agent.cost import ModelPrice, create_pricing_table
from repocompass.agent.context import (
    ContextSection,
    ContextSelectionDecision,
    ContextSelectionOutcome,
    ContextSelectionReason,
    ContextSourceKind,
    ProvenanceKind,
    ProvenanceLink,
    RawArtifactKind,
    RepositoryLocation,
    VerificationStatus,
    create_context_build_report,
    create_context_item,
    create_context_pack,
    create_context_source,
    create_raw_artifact_metadata,
)
from repocompass.agent.usage import (
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
    UsageRecord,
    create_model_invocation_record,
)
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
    RawStoreConflictError,
    RawStoreCorruptionError,
    RawStoreIntegrityError,
    open_raw_run_store,
)

REPOSITORY_ID = "a" * 64
COMMIT_SHA = "b" * 40
OBJECTIVE_DIGEST = "c" * 64


def _attempt(
    run_id: str = "run-1",
    *,
    thread_id: str = "thread-1",
    status: AttemptStatus = AttemptStatus.RUNNING,
    finished_at_ms: int | None = None,
) -> RawRunAttempt:
    return RawRunAttempt(
        run_id=run_id,
        thread_id=thread_id,
        turn_number=1,
        execution_mode=AttemptExecutionMode.START,
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
        objective_digest=OBJECTIVE_DIGEST,
        status=status,
        started_at_ms=1_000,
        finished_at_ms=finished_at_ms,
    )


def _artifact(
    content: bytes = b"def authenticate():\n    return True\n",
    *,
    kind: RawArtifactKind = RawArtifactKind.TOOL_OUTPUT,
):
    metadata = create_raw_artifact_metadata(
        kind=kind,
        media_type="text/plain",
        content=content,
    )
    return metadata, content


def _context_pack(artifact_id: str):
    source = create_context_source(
        kind=ContextSourceKind.SOURCE_EXCERPT,
        raw_artifact_id=artifact_id,
        location=RepositoryLocation(
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
            path="src/auth.py",
            line_start=1,
            line_end=2,
            symbol="authenticate",
        ),
        provenance=(
            ProvenanceLink(kind=ProvenanceKind.PLAN_STEP, id="inspect-auth"),
            ProvenanceLink(kind=ProvenanceKind.TOOL_CALL, id="call-read-auth"),
        ),
    )
    item = create_context_item(
        source=source,
        section=ContextSection.EVIDENCE,
        verification_status=VerificationStatus.VERIFIED,
        content="src/auth.py:1-2 defines authenticate.",
        estimated_tokens=10,
        priority=90,
    )
    report = create_context_build_report(
        budget_tokens=20,
        decisions=(
            ContextSelectionDecision(
                item_id=item.item_id,
                outcome=ContextSelectionOutcome.SELECTED,
                reason=ContextSelectionReason.VERIFIED_EVIDENCE,
                estimated_tokens=item.estimated_tokens,
                selected_position=0,
            ),
        ),
    )
    return create_context_pack(report=report, items=(item,))


def _invocation(
    *,
    input_artifact_id: str,
    output_artifact_id: str,
    context_pack_id: str,
    latency_ms: int = 25,
):
    return create_model_invocation_record(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        sequence=0,
        node_name="evaluate_candidate",
        role=ModelRole.EVALUATOR,
        provider="openai",
        model="gpt-5-mini",
        status=ModelInvocationStatus.SUCCEEDED,
        latency_ms=latency_ms,
        usage=UsageRecord(
            availability=UsageAvailability.REPORTED,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cached_input_tokens=10,
        ),
        plan_step_id="inspect-auth",
        context_pack_id=context_pack_id,
        input_artifact_id=input_artifact_id,
        output_artifact_id=output_artifact_id,
    )


def test_attempt_contract_requires_canonical_lifecycle_timestamps() -> None:
    with pytest.raises(ValidationError, match="finished_at_ms"):
        _attempt(status=AttemptStatus.COMPLETED)
    with pytest.raises(ValidationError, match="finished_at_ms"):
        _attempt(status=AttemptStatus.RUNNING, finished_at_ms=2_000)
    with pytest.raises(ValidationError, match="started_at_ms"):
        _attempt(
            status=AttemptStatus.COMPLETED,
            finished_at_ms=999,
        )


def test_attempt_round_trip_is_idempotent_and_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    attempt = _attempt()

    with open_raw_run_store(database_path) as store:
        assert store.record_attempt(attempt) is True
        assert store.record_attempt(attempt) is False
        assert store.get_attempt(attempt.run_id) == attempt

        conflicting = RawRunAttempt.model_validate(
            {
                **attempt.model_dump(mode="python"),
                "objective_digest": "d" * 64,
            }
        )
        with pytest.raises(RawStoreConflictError, match="attempt"):
            store.record_attempt(conflicting)


def test_list_attempts_is_bounded_ordered_and_filterable(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"
    older = _attempt(run_id="run-a", thread_id="thread-a")
    newer = RawRunAttempt.model_validate(
        {
            **_attempt(run_id="run-b", thread_id="thread-b").model_dump(mode="python"),
            "started_at_ms": 2_000,
        }
    )

    with open_raw_run_store(database_path) as store:
        store.record_attempt(older)
        store.record_attempt(newer)

        assert store.list_attempts(max_results=1) == (newer,)
        assert store.list_attempts(
            max_results=10,
            thread_id="thread-a",
        ) == (older,)
        with pytest.raises(ValueError, match="max_results"):
            store.list_attempts(max_results=0)
        with pytest.raises(ValueError, match="thread_id"):
            store.list_attempts(max_results=1, thread_id="../outside")


def test_finish_attempt_is_one_way_and_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        finished = store.finish_attempt(
            "run-1",
            status=AttemptStatus.COMPLETED,
            finished_at_ms=2_000,
        )

        assert finished.status is AttemptStatus.COMPLETED
        assert finished.finished_at_ms == 2_000
        assert (
            store.finish_attempt(
                "run-1",
                status=AttemptStatus.COMPLETED,
                finished_at_ms=2_000,
            )
            == finished
        )
        with pytest.raises(RawStoreConflictError, match="terminal"):
            store.finish_attempt(
                "run-1",
                status=AttemptStatus.FAILED,
                finished_at_ms=2_001,
            )


def test_raw_artifact_round_trip_and_attempt_link_are_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    metadata, content = _artifact()

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        assert store.put_artifact("run-1", metadata, content) is True
        assert store.put_artifact("run-1", metadata, content) is False

        stored = store.get_artifact(metadata.artifact_id)
        assert stored is not None
        assert stored.metadata == metadata
        assert stored.content == content
        assert store.list_artifacts("run-1") == (metadata,)


def test_artifact_content_mismatch_rolls_back_global_and_attempt_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    metadata, _ = _artifact()

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        with pytest.raises(RawStoreIntegrityError, match="metadata"):
            store.put_artifact("run-1", metadata, b"tampered")

        assert store.get_artifact(metadata.artifact_id) is None
        assert store.list_artifacts("run-1") == ()


def test_missing_attempt_artifact_write_rolls_back_content_row(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"
    metadata, content = _artifact()

    with open_raw_run_store(database_path) as store:
        with pytest.raises(RawStoreIntegrityError, match="attempt"):
            store.put_artifact("missing-run", metadata, content)
        assert store.get_artifact(metadata.artifact_id) is None


def test_context_pack_requires_every_raw_source_linked_to_attempt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    metadata, content = _artifact()
    pack = _context_pack(metadata.artifact_id)

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        with pytest.raises(RawStoreIntegrityError, match="artifact"):
            store.record_context_pack("run-1", sequence=0, pack=pack)

        store.put_artifact("run-1", metadata, content)
        assert store.record_context_pack("run-1", sequence=0, pack=pack) is True
        assert store.record_context_pack("run-1", sequence=0, pack=pack) is False
        assert store.get_context_pack("run-1", pack.pack_id) == pack
        assert store.list_context_packs("run-1") == (pack,)


def test_context_pack_sequence_or_identity_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    first_metadata, first_content = _artifact()
    second_metadata, second_content = _artifact(b"different source")
    first_pack = _context_pack(first_metadata.artifact_id)
    second_pack = _context_pack(second_metadata.artifact_id)

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        store.put_artifact("run-1", first_metadata, first_content)
        store.put_artifact("run-1", second_metadata, second_content)
        store.record_context_pack("run-1", sequence=0, pack=first_pack)

        with pytest.raises(RawStoreConflictError, match="context"):
            store.record_context_pack("run-1", sequence=0, pack=second_pack)


def test_model_invocation_round_trip_requires_full_provenance_chain(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    input_metadata, input_content = _artifact(
        b"model input",
        kind=RawArtifactKind.MODEL_INPUT,
    )
    output_metadata, output_content = _artifact(
        b"model output",
        kind=RawArtifactKind.MODEL_OUTPUT,
    )
    context_metadata, context_content = _artifact(b"context source")
    pack = _context_pack(context_metadata.artifact_id)
    invocation = _invocation(
        input_artifact_id=input_metadata.artifact_id,
        output_artifact_id=output_metadata.artifact_id,
        context_pack_id=pack.pack_id,
    )

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        store.put_artifact("run-1", input_metadata, input_content)
        store.put_artifact("run-1", output_metadata, output_content)
        store.put_artifact("run-1", context_metadata, context_content)
        store.record_context_pack("run-1", sequence=0, pack=pack)

        assert store.record_model_invocation(invocation) is True
        assert store.record_model_invocation(invocation) is False
        assert store.get_model_invocation(invocation.invocation_id) == invocation
        assert store.list_model_invocations("run-1") == (invocation,)
        report = store.get_attempt_usage_cost_report(
            "run-1",
            create_pricing_table(
                version="test-v1",
                effective_from="2026-08-24",
                entries=(
                    ModelPrice(
                        provider="openai",
                        model="gpt-5-mini",
                        input_usd_per_million_tokens="2",
                        output_usd_per_million_tokens="8",
                    ),
                ),
            ),
        )
        assert report is not None
        assert report.breakdowns[0].cost.estimated_cost_usd == "0.00036"


@pytest.mark.parametrize("missing", ["input", "output", "context"])
def test_model_invocation_missing_provenance_rolls_back(
    tmp_path: Path,
    missing: str,
) -> None:
    database_path = tmp_path / f"{missing}.sqlite"
    input_metadata, input_content = _artifact(
        b"model input",
        kind=RawArtifactKind.MODEL_INPUT,
    )
    output_metadata, output_content = _artifact(
        b"model output",
        kind=RawArtifactKind.MODEL_OUTPUT,
    )
    context_metadata, context_content = _artifact(b"context source")
    pack = _context_pack(context_metadata.artifact_id)
    invocation = _invocation(
        input_artifact_id=input_metadata.artifact_id,
        output_artifact_id=output_metadata.artifact_id,
        context_pack_id=pack.pack_id,
    )

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        store.put_artifact("run-1", context_metadata, context_content)
        if missing != "input":
            store.put_artifact("run-1", input_metadata, input_content)
        if missing != "output":
            store.put_artifact("run-1", output_metadata, output_content)
        if missing != "context":
            store.record_context_pack("run-1", sequence=0, pack=pack)

        with pytest.raises(RawStoreIntegrityError, match="provenance"):
            store.record_model_invocation(invocation)
        assert store.list_model_invocations("run-1") == ()


@pytest.mark.parametrize(
    ("wrong_role", "expected_kind"),
    [
        ("input", "MODEL_INPUT"),
        ("output", "MODEL_OUTPUT"),
    ],
)
def test_model_invocation_rejects_wrong_artifact_kind(
    tmp_path: Path,
    wrong_role: str,
    expected_kind: str,
) -> None:
    database_path = tmp_path / f"wrong-{wrong_role}.sqlite"
    input_kind = (
        RawArtifactKind.TOOL_OUTPUT
        if wrong_role == "input"
        else RawArtifactKind.MODEL_INPUT
    )
    output_kind = (
        RawArtifactKind.TOOL_OUTPUT
        if wrong_role == "output"
        else RawArtifactKind.MODEL_OUTPUT
    )
    input_metadata, input_content = _artifact(b"model input", kind=input_kind)
    output_metadata, output_content = _artifact(b"model output", kind=output_kind)
    context_metadata, context_content = _artifact(b"context source")
    pack = _context_pack(context_metadata.artifact_id)
    invocation = _invocation(
        input_artifact_id=input_metadata.artifact_id,
        output_artifact_id=output_metadata.artifact_id,
        context_pack_id=pack.pack_id,
    )

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        store.put_artifact("run-1", input_metadata, input_content)
        store.put_artifact("run-1", output_metadata, output_content)
        store.put_artifact("run-1", context_metadata, context_content)
        store.record_context_pack("run-1", sequence=0, pack=pack)

        with pytest.raises(RawStoreIntegrityError, match=expected_kind):
            store.record_model_invocation(invocation)
        assert store.list_model_invocations("run-1") == ()


def test_model_invocation_conflicting_rewrite_is_rejected(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"
    input_metadata, input_content = _artifact(
        b"model input",
        kind=RawArtifactKind.MODEL_INPUT,
    )
    output_metadata, output_content = _artifact(
        b"model output",
        kind=RawArtifactKind.MODEL_OUTPUT,
    )
    context_metadata, context_content = _artifact(b"context source")
    pack = _context_pack(context_metadata.artifact_id)
    first = _invocation(
        input_artifact_id=input_metadata.artifact_id,
        output_artifact_id=output_metadata.artifact_id,
        context_pack_id=pack.pack_id,
        latency_ms=25,
    )
    conflicting = _invocation(
        input_artifact_id=input_metadata.artifact_id,
        output_artifact_id=output_metadata.artifact_id,
        context_pack_id=pack.pack_id,
        latency_ms=99,
    )

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        store.put_artifact("run-1", input_metadata, input_content)
        store.put_artifact("run-1", output_metadata, output_content)
        store.put_artifact("run-1", context_metadata, context_content)
        store.record_context_pack("run-1", sequence=0, pack=pack)
        store.record_model_invocation(first)

        with pytest.raises(RawStoreConflictError, match="invocation"):
            store.record_model_invocation(conflicting)


def test_invocation_attempt_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"
    input_metadata, input_content = _artifact(
        b"model input",
        kind=RawArtifactKind.MODEL_INPUT,
    )
    output_metadata, output_content = _artifact(
        b"model output",
        kind=RawArtifactKind.MODEL_OUTPUT,
    )
    context_metadata, context_content = _artifact(b"context source")
    pack = _context_pack(context_metadata.artifact_id)
    invocation = _invocation(
        input_artifact_id=input_metadata.artifact_id,
        output_artifact_id=output_metadata.artifact_id,
        context_pack_id=pack.pack_id,
    )
    payload = invocation.model_dump(mode="python")
    mismatched = type(invocation).model_validate(
        {**payload, "thread_id": "another-thread"}
    )

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())
        store.put_artifact("run-1", input_metadata, input_content)
        store.put_artifact("run-1", output_metadata, output_content)
        store.put_artifact("run-1", context_metadata, context_content)
        store.record_context_pack("run-1", sequence=0, pack=pack)

        with pytest.raises(RawStoreIntegrityError, match="attempt"):
            store.record_model_invocation(mismatched)


def test_delete_attempt_cascades_records_and_collects_only_orphan_artifacts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "raw.sqlite"
    metadata, content = _artifact()

    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt("run-1", thread_id="thread-1"))
        store.record_attempt(_attempt("run-2", thread_id="thread-2"))
        store.put_artifact("run-1", metadata, content)
        store.put_artifact("run-2", metadata, content)

        assert store.delete_attempt("run-1") is True
        assert store.get_attempt("run-1") is None
        assert store.get_artifact(metadata.artifact_id) is not None
        assert store.list_artifacts("run-2") == (metadata,)

        assert store.delete_attempt("run-2") is True
        assert store.get_artifact(metadata.artifact_id) is None
        assert store.delete_attempt("run-2") is False


def test_concurrent_connections_write_without_losing_artifacts(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"
    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())

    def write_artifact(index: int) -> str:
        content = f"artifact-{index}".encode()
        metadata, _ = _artifact(content)
        with open_raw_run_store(database_path) as store:
            store.put_artifact("run-1", metadata, content)
        return metadata.artifact_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        artifact_ids = tuple(executor.map(write_artifact, range(12)))

    with open_raw_run_store(database_path) as store:
        assert len(set(artifact_ids)) == 12
        assert {item.artifact_id for item in store.list_artifacts("run-1")} == set(
            artifact_ids
        )


def test_corrupted_record_json_fails_closed_on_read(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"
    with open_raw_run_store(database_path) as store:
        store.record_attempt(_attempt())

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE attempts SET record_json = ? WHERE run_id = ?",
            ("{}", "run-1"),
        )

    with open_raw_run_store(database_path) as store:
        with pytest.raises(RawStoreCorruptionError, match="attempt"):
            store.get_attempt("run-1")
