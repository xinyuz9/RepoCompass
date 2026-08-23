"""Contract tests for M3 tool safety and no-progress boundaries."""

import pytest
from pydantic import ValidationError

from repocompass.agent.checkpoint import RepoCompassCheckpointSerializer
from repocompass.agent.tool_adapters import REPOSITORY_TOOLS
from repocompass.tools.contracts import (
    NoProgressAction,
    NoProgressDecision,
    NoProgressReason,
    ProgressSnapshot,
    ToolAuditRecord,
    ToolCallFingerprint,
    ToolCallRequest,
    ToolCapability,
    ToolDecisionOutcome,
    ToolDecisionReason,
    ToolDescriptor,
    ToolPolicyDecision,
    ToolRiskLevel,
    ToolSideEffect,
    fingerprint_progress_snapshot,
    fingerprint_tool_call,
)
from repocompass.tools.repository_tools import REPOSITORY_TOOL_DESCRIPTORS

REPOSITORY_ID = "a" * 64
OTHER_REPOSITORY_ID = "c" * 64
COMMIT_SHA = "b" * 40
RESULT_DIGEST = "d" * 64


def _descriptor(**overrides: object) -> ToolDescriptor:
    values: dict[str, object] = {
        "name": "read_code",
        "capabilities": (ToolCapability.FILESYSTEM_READ,),
        "side_effect": ToolSideEffect.READ_ONLY,
        "risk_level": ToolRiskLevel.LOW,
        "timeout_ms": 5_000,
        "max_output_bytes": 65_536,
        "cacheable": True,
    }
    values.update(overrides)
    return ToolDescriptor.model_validate(values)


def _request(**overrides: object) -> ToolCallRequest:
    values: dict[str, object] = {
        "call_id": "call-1",
        "tool_name": "search_text",
        "arguments": {
            "query": "authenticate",
            "globs": ["*.py"],
            "max_results": 25,
        },
        "repository_id": REPOSITORY_ID,
        "commit_sha": COMMIT_SHA,
    }
    values.update(overrides)
    return ToolCallRequest.model_validate(values)


def _fingerprint() -> ToolCallFingerprint:
    return fingerprint_tool_call(_request())


def _decision(
    outcome: ToolDecisionOutcome = ToolDecisionOutcome.ALLOWED,
    reason: ToolDecisionReason = ToolDecisionReason.ALLOWED,
) -> ToolPolicyDecision:
    return ToolPolicyDecision(outcome=outcome, reason=reason)


def test_tool_safety_vocabularies_are_closed() -> None:
    assert set(ToolCapability) == {
        ToolCapability.FILESYSTEM_READ,
        ToolCapability.NETWORK,
        ToolCapability.PROCESS,
        ToolCapability.FILESYSTEM_WRITE,
        ToolCapability.DESTRUCTIVE,
    }
    assert set(ToolSideEffect) == {
        ToolSideEffect.READ_ONLY,
        ToolSideEffect.WRITE,
        ToolSideEffect.DESTRUCTIVE,
    }
    assert set(ToolRiskLevel) == {
        ToolRiskLevel.LOW,
        ToolRiskLevel.ELEVATED,
        ToolRiskLevel.HIGH,
        ToolRiskLevel.CRITICAL,
    }
    assert set(ToolDecisionOutcome) == {
        ToolDecisionOutcome.ALLOWED,
        ToolDecisionOutcome.DENIED,
        ToolDecisionOutcome.CACHED,
        ToolDecisionOutcome.DEDUPLICATED,
        ToolDecisionOutcome.NO_PROGRESS,
    }
    assert set(ToolDecisionReason) == {
        ToolDecisionReason.ALLOWED,
        ToolDecisionReason.UNKNOWN_TOOL,
        ToolDecisionReason.INVALID_ARGUMENTS,
        ToolDecisionReason.PATH_ESCAPE,
        ToolDecisionReason.CAPABILITY_DENIED,
        ToolDecisionReason.BUDGET_EXHAUSTED,
        ToolDecisionReason.DUPLICATE_CALL,
        ToolDecisionReason.CACHE_HIT,
        ToolDecisionReason.TIMEOUT,
        ToolDecisionReason.OVERSIZED_OUTPUT,
        ToolDecisionReason.EXECUTION_ERROR,
        ToolDecisionReason.NO_PROGRESS,
    }


def test_tool_descriptor_is_strict_versioned_and_bounded() -> None:
    descriptor = _descriptor()

    assert set(ToolDescriptor.model_fields) == {
        "schema_version",
        "name",
        "capabilities",
        "side_effect",
        "risk_level",
        "timeout_ms",
        "max_output_bytes",
        "cacheable",
    }
    assert descriptor.schema_version == 1

    with pytest.raises(ValidationError):
        _descriptor(timeout_ms="5000")
    with pytest.raises(ValidationError):
        ToolDescriptor.model_validate(
            {**descriptor.model_dump(mode="json"), "authorization": "secret"}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "capabilities": (
                ToolCapability.FILESYSTEM_READ,
                ToolCapability.FILESYSTEM_READ,
            )
        },
        {
            "capabilities": (
                ToolCapability.PROCESS,
                ToolCapability.FILESYSTEM_READ,
            ),
            "risk_level": ToolRiskLevel.ELEVATED,
        },
        {
            "capabilities": (ToolCapability.FILESYSTEM_WRITE,),
            "side_effect": ToolSideEffect.READ_ONLY,
            "risk_level": ToolRiskLevel.HIGH,
        },
        {
            "capabilities": (ToolCapability.FILESYSTEM_WRITE,),
            "side_effect": ToolSideEffect.WRITE,
            "risk_level": ToolRiskLevel.HIGH,
            "cacheable": True,
        },
        {
            "capabilities": (ToolCapability.DESTRUCTIVE,),
            "side_effect": ToolSideEffect.DESTRUCTIVE,
            "risk_level": ToolRiskLevel.HIGH,
            "cacheable": False,
        },
    ],
)
def test_tool_descriptor_rejects_ambiguous_or_unsafe_metadata(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _descriptor(**overrides)


def test_repository_tools_declare_only_bounded_filesystem_reads() -> None:
    tool_names = {repository_tool.name for repository_tool in REPOSITORY_TOOLS}

    assert set(REPOSITORY_TOOL_DESCRIPTORS) == tool_names
    for descriptor in REPOSITORY_TOOL_DESCRIPTORS.values():
        assert descriptor.capabilities == (ToolCapability.FILESYSTEM_READ,)
        assert descriptor.side_effect is ToolSideEffect.READ_ONLY
        assert descriptor.risk_level is ToolRiskLevel.LOW
        assert descriptor.cacheable is True
        assert descriptor.timeout_ms > 0
        assert descriptor.max_output_bytes > 0


def test_tool_call_request_accepts_only_stable_json_arguments() -> None:
    request = _request()

    assert set(ToolCallRequest.model_fields) == {
        "schema_version",
        "call_id",
        "tool_name",
        "arguments",
        "repository_id",
        "commit_sha",
    }
    assert request.schema_version == 1

    with pytest.raises(ValidationError):
        _request(arguments={"max_results": float("nan")})
    with pytest.raises(ValidationError):
        _request(arguments={"max_results": float("inf")})
    with pytest.raises(ValidationError):
        _request(arguments={"globs": ("*.py",)})
    with pytest.raises(ValidationError):
        ToolCallRequest.model_validate(
            {**request.model_dump(mode="json"), "repo_path": "/private/repo"}
        )
    with pytest.raises(ValidationError, match="safe integer"):
        _request(arguments={"max_results": 2**53})
    with pytest.raises(ValidationError, match="Unicode"):
        _request(arguments={"query": "\ud800"})
    with pytest.raises(ValidationError, match="byte limit"):
        _request(arguments={"query": "x" * 65_536})


def test_tool_call_request_rejects_excessive_json_nesting() -> None:
    nested: object = "leaf"
    for _ in range(33):
        nested = [nested]

    with pytest.raises(ValidationError, match="nesting"):
        _request(arguments={"nested": nested})


def test_tool_call_request_round_trips_through_strict_json_schema() -> None:
    request = _request(arguments={"query": "身份验证", "nested": {"ok": True}})

    restored = ToolCallRequest.model_validate_json(request.model_dump_json())

    assert restored == request


def test_tool_call_fingerprint_is_canonical_and_ignores_call_identity() -> None:
    first = _request(
        call_id="call-first",
        arguments={
            "query": "authenticate",
            "options": {"case_sensitive": False, "limit": 25},
        },
    )
    reordered = _request(
        call_id="call-second",
        arguments={
            "options": {"limit": 25, "case_sensitive": False},
            "query": "authenticate",
        },
    )

    first_fingerprint = fingerprint_tool_call(first)
    reordered_fingerprint = fingerprint_tool_call(reordered)

    assert first_fingerprint == reordered_fingerprint
    assert first_fingerprint.schema_version == 1
    assert first_fingerprint.algorithm == "sha256"
    assert len(first_fingerprint.digest) == 64


def test_tool_call_fingerprint_has_a_versioned_cross_process_fixture() -> None:
    request = _request(
        call_id="call-fixture",
        arguments={
            "globs": ["*.py"],
            "max_results": 25,
            "query": "authenticate",
        },
    )

    assert (
        fingerprint_tool_call(request).digest
        == "9206d62cd8bab2e1c194de12940e0a02890ca827dd3375bd474fdd388e9951a6"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tool_name": "read_code"},
        {"arguments": {"query": "authorize"}},
        {"repository_id": OTHER_REPOSITORY_ID},
        {"commit_sha": "c" * 40},
        {"commit_sha": None},
    ],
)
def test_tool_call_fingerprint_covers_every_execution_identity_field(
    overrides: dict[str, object],
) -> None:
    assert fingerprint_tool_call(_request(**overrides)) != _fingerprint()


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ToolDecisionOutcome.ALLOWED, ToolDecisionReason.ALLOWED),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.UNKNOWN_TOOL),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.INVALID_ARGUMENTS),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.PATH_ESCAPE),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.CAPABILITY_DENIED),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.BUDGET_EXHAUSTED),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.TIMEOUT),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.OVERSIZED_OUTPUT),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.EXECUTION_ERROR),
        (ToolDecisionOutcome.CACHED, ToolDecisionReason.CACHE_HIT),
        (
            ToolDecisionOutcome.DEDUPLICATED,
            ToolDecisionReason.DUPLICATE_CALL,
        ),
        (ToolDecisionOutcome.NO_PROGRESS, ToolDecisionReason.NO_PROGRESS),
    ],
)
def test_tool_policy_decision_accepts_only_canonical_outcome_reason_pairs(
    outcome: ToolDecisionOutcome,
    reason: ToolDecisionReason,
) -> None:
    decision = _decision(outcome, reason)

    restored = ToolPolicyDecision.model_validate_json(decision.model_dump_json())

    assert restored == decision


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ToolDecisionOutcome.ALLOWED, ToolDecisionReason.CACHE_HIT),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.DUPLICATE_CALL),
        (ToolDecisionOutcome.CACHED, ToolDecisionReason.ALLOWED),
        (ToolDecisionOutcome.DEDUPLICATED, ToolDecisionReason.CACHE_HIT),
        (ToolDecisionOutcome.NO_PROGRESS, ToolDecisionReason.BUDGET_EXHAUSTED),
    ],
)
def test_tool_policy_decision_rejects_mismatched_outcome_reason(
    outcome: ToolDecisionOutcome,
    reason: ToolDecisionReason,
) -> None:
    with pytest.raises(ValidationError, match="outcome and reason"):
        _decision(outcome, reason)


def test_tool_audit_record_contains_only_bounded_metadata() -> None:
    audit = ToolAuditRecord(
        sequence=3,
        run_id="run-1",
        call_id="call-1",
        tool_name="search_text",
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
        fingerprint=_fingerprint(),
        decision=_decision(),
        latency_ms=12,
        output_bytes=128,
        result_digest=RESULT_DIGEST,
    )

    assert set(ToolAuditRecord.model_fields) == {
        "schema_version",
        "sequence",
        "run_id",
        "call_id",
        "tool_name",
        "repository_id",
        "commit_sha",
        "fingerprint",
        "decision",
        "latency_ms",
        "output_bytes",
        "result_digest",
    }
    assert ToolAuditRecord.model_validate_json(audit.model_dump_json()) == audit
    assert "arguments" not in audit.model_dump(mode="json")
    assert "output" not in audit.model_dump(mode="json")


@pytest.mark.parametrize(
    "secret_field",
    ["authorization", "api_key", "token", "password", "reasoning_content"],
)
def test_tool_audit_record_rejects_secret_or_unbounded_fields(
    secret_field: str,
) -> None:
    values = {
        "sequence": 0,
        "run_id": "run-1",
        "call_id": "call-1",
        "tool_name": "search_text",
        "repository_id": REPOSITORY_ID,
        "commit_sha": COMMIT_SHA,
        "fingerprint": _fingerprint(),
        "decision": _decision(),
        "latency_ms": 1,
        "output_bytes": None,
        "result_digest": None,
        secret_field: "super-secret-value",
    }

    with pytest.raises(ValidationError):
        ToolAuditRecord.model_validate(values)


def test_progress_snapshot_requires_canonical_unique_fingerprints() -> None:
    snapshot = ProgressSnapshot(
        active_step_id="locate-auth",
        evidence_fingerprints=("1" * 64, "2" * 64),
        successful_tool_fingerprints=("3" * 64, "4" * 64),
        evaluator_request_digest="5" * 64,
        plan_revision_count=1,
    )

    assert ProgressSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot

    with pytest.raises(ValidationError, match="canonical"):
        ProgressSnapshot(
            active_step_id="locate-auth",
            evidence_fingerprints=("2" * 64, "1" * 64),
        )
    with pytest.raises(ValidationError, match="canonical"):
        ProgressSnapshot(
            active_step_id="locate-auth",
            successful_tool_fingerprints=("3" * 64, "3" * 64),
        )


def test_progress_snapshot_fingerprint_covers_progress_signals() -> None:
    baseline = ProgressSnapshot(active_step_id="locate-auth")
    with_evidence = ProgressSnapshot(
        active_step_id="locate-auth",
        evidence_fingerprints=("1" * 64,),
    )

    assert fingerprint_progress_snapshot(baseline) == fingerprint_progress_snapshot(
        ProgressSnapshot(active_step_id="locate-auth")
    )
    assert fingerprint_progress_snapshot(baseline) != fingerprint_progress_snapshot(
        with_evidence
    )


@pytest.mark.parametrize(
    ("action", "reason", "stalled_rounds", "threshold"),
    [
        (
            NoProgressAction.CONTINUE,
            NoProgressReason.PROGRESS_OBSERVED,
            0,
            3,
        ),
        (
            NoProgressAction.CONTINUE,
            NoProgressReason.BELOW_THRESHOLD,
            2,
            3,
        ),
        (NoProgressAction.STOP, NoProgressReason.NO_PROGRESS_LIMIT, 3, 3),
        (NoProgressAction.REPLAN, NoProgressReason.EVALUATOR_REPLAN, 1, 3),
    ],
)
def test_no_progress_decision_is_deterministic(
    action: NoProgressAction,
    reason: NoProgressReason,
    stalled_rounds: int,
    threshold: int,
) -> None:
    decision = NoProgressDecision(
        action=action,
        reason=reason,
        consecutive_stalled_rounds=stalled_rounds,
        threshold=threshold,
        snapshot_digest="f" * 64,
    )

    assert (
        NoProgressDecision.model_validate_json(decision.model_dump_json()) == decision
    )


def test_no_progress_decision_rejects_inconsistent_action_or_threshold() -> None:
    with pytest.raises(ValidationError):
        NoProgressDecision(
            action=NoProgressAction.STOP,
            reason=NoProgressReason.NO_PROGRESS_LIMIT,
            consecutive_stalled_rounds=2,
            threshold=3,
            snapshot_digest="f" * 64,
        )


def test_tool_contracts_survive_repeated_narrow_msgpack_round_trips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptor = _descriptor()
    request = _request()
    fingerprint = fingerprint_tool_call(request)
    decision = _decision()
    audit = ToolAuditRecord(
        sequence=0,
        run_id="run-1",
        call_id=request.call_id,
        tool_name=request.tool_name,
        repository_id=request.repository_id,
        commit_sha=request.commit_sha,
        fingerprint=fingerprint,
        decision=decision,
        latency_ms=2,
        output_bytes=32,
        result_digest=RESULT_DIGEST,
    )
    progress = ProgressSnapshot(
        active_step_id="locate-auth",
        successful_tool_fingerprints=(fingerprint.digest,),
    )
    no_progress = NoProgressDecision(
        action=NoProgressAction.CONTINUE,
        reason=NoProgressReason.PROGRESS_OBSERVED,
        consecutive_stalled_rounds=0,
        threshold=3,
        snapshot_digest=fingerprint_progress_snapshot(progress),
    )
    expected = {
        "descriptor": descriptor,
        "request": request,
        "audit": audit,
        "progress": progress,
        "no_progress": no_progress,
    }
    serializer = RepoCompassCheckpointSerializer()
    current: object = expected

    for _ in range(3):
        current = serializer.loads_typed(serializer.dumps_typed(current))

    assert current == expected
    assert isinstance(current, dict)
    assert isinstance(current["descriptor"], ToolDescriptor)
    assert isinstance(current["request"], ToolCallRequest)
    restored_audit = current["audit"]
    assert isinstance(restored_audit, ToolAuditRecord)
    assert isinstance(restored_audit.fingerprint, ToolCallFingerprint)
    assert isinstance(restored_audit.decision, ToolPolicyDecision)
    assert isinstance(current["progress"], ProgressSnapshot)
    assert isinstance(current["no_progress"], NoProgressDecision)
    assert "Blocked deserialization" not in caplog.text
    with pytest.raises(ValidationError):
        NoProgressDecision(
            action=NoProgressAction.CONTINUE,
            reason=NoProgressReason.NO_PROGRESS_LIMIT,
            consecutive_stalled_rounds=3,
            threshold=3,
            snapshot_digest="f" * 64,
        )
