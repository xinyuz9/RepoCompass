"""Unit and security tests for the M3.1 Tool Gateway core."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import repocompass.tools.gateway as gateway_module
from repocompass.tools.contracts import (
    ToolCapability,
    ToolDecisionOutcome,
    ToolDecisionReason,
    ToolDescriptor,
    ToolRiskLevel,
    ToolSideEffect,
)
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    ToolRegistration,
    ToolRegistry,
    create_repository_tool_registry,
)
from repocompass.tools.repository_tools import (
    REPOSITORY_TOOL_DESCRIPTORS,
    ListTreeArguments,
)


def _return_normalized_text(
    _root: Path, _arguments: BaseModel, _repository_id: str, _commit_sha: str | None
) -> str:
    return "first\r\nsecond\rthird"


def _return_oversized_text(
    _root: Path, _arguments: BaseModel, _repository_id: str, _commit_sha: str | None
) -> str:
    return "界" * 3


def _return_invalid_unicode(
    _root: Path, _arguments: BaseModel, _repository_id: str, _commit_sha: str | None
) -> str:
    return "\ud800"


def _sleep_past_deadline(
    _root: Path, _arguments: BaseModel, _repository_id: str, _commit_sha: str | None
) -> str:
    time.sleep(0.25)
    return "late"


def _raise_secret_error(
    _root: Path, _arguments: BaseModel, _repository_id: str, _commit_sha: str | None
) -> str:
    raise RuntimeError("authorization=Bearer super-secret")


def _write_forbidden_marker(
    root: Path, _arguments: BaseModel, _repository_id: str, _commit_sha: str | None
) -> str:
    (root / "forbidden.txt").write_text("should not execute", encoding="utf-8")
    return "written"


def _descriptor(
    name: str,
    *,
    capabilities: tuple[ToolCapability, ...] = (ToolCapability.FILESYSTEM_READ,),
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY,
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    timeout_ms: int = 1_000,
    max_output_bytes: int = 1_024,
    cacheable: bool = True,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        capabilities=capabilities,
        side_effect=side_effect,
        risk_level=risk_level,
        timeout_ms=timeout_ms,
        max_output_bytes=max_output_bytes,
        cacheable=cacheable,
    )


def _gateway_for(
    descriptor: ToolDescriptor,
    executor,
    *,
    repository_path_field: str | None = None,
) -> ToolGateway:
    registration = ToolRegistration(
        descriptor=descriptor,
        arguments_model=ListTreeArguments,
        executor=executor,
        repository_path_field=repository_path_field,
    )
    return ToolGateway(ToolRegistry((registration,)))


def _execute(
    gateway: ToolGateway,
    context: ToolExecutionContext,
    *,
    tool_name: str,
    arguments: dict[str, object],
    call_id: str = "call-1",
    available_budget: int = 1,
    audit_log: ToolAuditLog | None = None,
):
    request = context.bind_call(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    log = audit_log or ToolAuditLog("run-1")
    result = gateway.execute(
        request,
        context=context,
        available_budget=available_budget,
        audit_log=log,
    )
    return result, log


def test_execution_context_binds_model_call_to_trusted_repository(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    restored = ToolExecutionContext(tmp_path)
    request = context.bind_call(
        call_id="call-1",
        tool_name="list_tree",
        arguments={},
    )

    assert context.repository_root == tmp_path.resolve()
    assert context.repository_id == restored.repository_id
    assert len(context.repository_id) == 64
    assert context.commit_sha is None
    assert request.repository_id == context.repository_id
    assert request.commit_sha == context.commit_sha


def test_execution_context_rejects_missing_or_file_repository(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        ToolExecutionContext(tmp_path / "missing")

    repository_file = tmp_path / "repository.txt"
    repository_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        ToolExecutionContext(repository_file)


def test_execution_context_rejects_unstable_ingress_values(tmp_path: Path) -> None:
    context = ToolExecutionContext(tmp_path)

    with pytest.raises(ValidationError):
        context.bind_call(
            call_id="call-1",
            tool_name="list_tree",
            arguments={"max_depth": float("nan")},
        )
    with pytest.raises(TypeError):
        context.bind_call(
            call_id="call-1",
            tool_name="list_tree",
            arguments={"max_depth": 3},
            repository_id="b" * 64,
        )


def test_gateway_denies_request_after_repository_commit_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_commit = {"value": "a" * 40}

    def discover_commit(_repository_root: Path) -> str:
        return current_commit["value"]

    monkeypatch.setattr(
        gateway_module,
        "_discover_commit_sha",
        discover_commit,
    )
    context = ToolExecutionContext(tmp_path)
    request = context.bind_call(
        call_id="call-1",
        tool_name="list_tree",
        arguments={},
    )
    current_commit["value"] = "b" * 40

    result = ToolGateway(create_repository_tool_registry()).execute(
        request,
        context=context,
        available_budget=1,
        audit_log=ToolAuditLog("run-1"),
    )

    assert result.decision.reason is ToolDecisionReason.CAPABILITY_DENIED
    assert result.budget_consumed == 0


def test_repository_registry_has_one_registration_per_descriptor() -> None:
    registry = create_repository_tool_registry()

    assert registry.names == tuple(REPOSITORY_TOOL_DESCRIPTORS)
    for name in registry.names:
        assert registry.require(name).descriptor == REPOSITORY_TOOL_DESCRIPTORS[name]


def test_registry_rejects_duplicate_names() -> None:
    descriptor = _descriptor("duplicate")
    registration = ToolRegistration(
        descriptor=descriptor,
        arguments_model=ListTreeArguments,
        executor=_return_normalized_text,
    )

    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry((registration, registration))


def test_registration_rejects_unknown_repository_path_field() -> None:
    with pytest.raises(ValueError, match="repository_path_field"):
        ToolRegistration(
            descriptor=_descriptor("invalid_path_field"),
            arguments_model=ListTreeArguments,
            executor=_return_normalized_text,
            repository_path_field="relative_path",
        )


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    [
        (
            "list_tree",
            {"max_depth": 2, "max_entries": 10},
            {"entries": ["app.py", "src/", "src/auth.py"]},
        ),
        (
            "search_text",
            {"query": "needle", "globs": ["*.py"], "max_results": 10},
            {"matches": [{"line": 1, "path": "app.py", "text": "needle = True"}]},
        ),
        (
            "read_code",
            {"relative_path": "src/auth.py", "start_line": 1, "end_line": 1},
            {
                "content": "1 | def authenticate():",
                "end_line": 1,
                "path": "src/auth.py",
                "start_line": 1,
            },
        ),
    ],
)
def test_gateway_executes_registered_repository_tools_with_bounded_json_output(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    expected: dict[str, object],
) -> None:
    (tmp_path / "app.py").write_text("needle = True\n", encoding="utf-8")
    source_directory = tmp_path / "src"
    source_directory.mkdir()
    (source_directory / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    context = ToolExecutionContext(tmp_path)

    result, audit_log = _execute(
        ToolGateway(create_repository_tool_registry()),
        context,
        tool_name=tool_name,
        arguments=arguments,
    )

    assert result.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert result.decision.reason is ToolDecisionReason.ALLOWED
    assert result.budget_consumed == 1
    assert json.loads(result.content) == expected
    assert result.audit_record == audit_log.records[0]
    assert result.audit_record.sequence == 0
    assert result.audit_record.output_bytes == len(result.content.encode("utf-8"))
    assert (
        result.audit_record.result_digest
        == hashlib.sha256(result.content.encode("utf-8")).hexdigest()
    )
    serialized_audit = result.audit_record.model_dump_json()
    assert "needle = True" not in serialized_audit
    assert "def authenticate" not in serialized_audit


def test_unknown_tool_is_denied_without_consuming_execution_budget(
    tmp_path: Path,
) -> None:
    result, audit_log = _execute(
        ToolGateway(create_repository_tool_registry()),
        ToolExecutionContext(tmp_path),
        tool_name="unknown_tool",
        arguments={},
    )

    assert result.decision.outcome is ToolDecisionOutcome.DENIED
    assert result.decision.reason is ToolDecisionReason.UNKNOWN_TOOL
    assert result.budget_consumed == 0
    assert result.audit_record.output_bytes is None
    assert result.audit_record.result_digest is None
    assert audit_log.records == (result.audit_record,)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list_tree", {"max_depth": "3"}),
        ("list_tree", {"unexpected": True}),
        ("search_text", {"query": ""}),
        (
            "read_code",
            {"relative_path": "app.py", "start_line": 3, "end_line": 2},
        ),
        (
            "read_code",
            {"relative_path": "app.py", "start_line": 1, "end_line": 402},
        ),
    ],
)
def test_invalid_tool_arguments_fail_closed_before_budget_reservation(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    result, _ = _execute(
        ToolGateway(create_repository_tool_registry()),
        ToolExecutionContext(tmp_path),
        tool_name=tool_name,
        arguments=arguments,
    )

    assert result.decision.outcome is ToolDecisionOutcome.DENIED
    assert result.decision.reason is ToolDecisionReason.INVALID_ARGUMENTS
    assert result.budget_consumed == 0


@pytest.mark.parametrize(
    "relative_path",
    ["/etc/passwd", "../outside.py", "src/../../outside.py"],
)
def test_gateway_denies_absolute_or_parent_escape_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    result, _ = _execute(
        ToolGateway(create_repository_tool_registry()),
        ToolExecutionContext(tmp_path),
        tool_name="read_code",
        arguments={
            "relative_path": relative_path,
            "start_line": 1,
            "end_line": 1,
        },
    )

    assert result.decision.reason is ToolDecisionReason.PATH_ESCAPE
    assert result.budget_consumed == 0


def test_gateway_denies_symlinks_even_when_the_target_is_internal(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.py"
    target.write_text("safe = True\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to(target)

    result, _ = _execute(
        ToolGateway(create_repository_tool_registry()),
        ToolExecutionContext(tmp_path),
        tool_name="read_code",
        arguments={"relative_path": "link.py", "start_line": 1, "end_line": 1},
    )

    assert result.decision.reason is ToolDecisionReason.CAPABILITY_DENIED
    assert result.budget_consumed == 0


def test_gateway_classifies_external_symlink_as_path_escape(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    (repository / "outside-link.py").symlink_to(outside)

    result, _ = _execute(
        ToolGateway(create_repository_tool_registry()),
        ToolExecutionContext(repository),
        tool_name="read_code",
        arguments={
            "relative_path": "outside-link.py",
            "start_line": 1,
            "end_line": 1,
        },
    )

    assert result.decision.reason is ToolDecisionReason.PATH_ESCAPE
    assert result.budget_consumed == 0


def test_gateway_denies_special_files_without_opening_them(tmp_path: Path) -> None:
    fifo_path = tmp_path / "stream.py"
    os.mkfifo(fifo_path)

    result, _ = _execute(
        ToolGateway(create_repository_tool_registry()),
        ToolExecutionContext(tmp_path),
        tool_name="read_code",
        arguments={"relative_path": "stream.py", "start_line": 1, "end_line": 1},
    )

    assert result.decision.reason is ToolDecisionReason.CAPABILITY_DENIED
    assert result.budget_consumed == 0


def test_gateway_denies_unapproved_capability_without_running_executor(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor(
        "network_read",
        capabilities=(ToolCapability.NETWORK,),
        risk_level=ToolRiskLevel.ELEVATED,
    )
    gateway = _gateway_for(descriptor, _write_forbidden_marker)

    result, _ = _execute(
        gateway,
        ToolExecutionContext(tmp_path),
        tool_name=descriptor.name,
        arguments={},
    )

    assert result.decision.reason is ToolDecisionReason.CAPABILITY_DENIED
    assert result.budget_consumed == 0
    assert not (tmp_path / "forbidden.txt").exists()


def test_gateway_denies_exhausted_budget_without_running_executor(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor("budgeted_read")
    gateway = _gateway_for(descriptor, _write_forbidden_marker)

    result, _ = _execute(
        gateway,
        ToolExecutionContext(tmp_path),
        tool_name=descriptor.name,
        arguments={},
        available_budget=0,
    )

    assert result.decision.reason is ToolDecisionReason.BUDGET_EXHAUSTED
    assert result.budget_consumed == 0
    assert not (tmp_path / "forbidden.txt").exists()


def test_gateway_rejects_invalid_trusted_budget(tmp_path: Path) -> None:
    context = ToolExecutionContext(tmp_path)
    request = context.bind_call(
        call_id="call-1",
        tool_name="list_tree",
        arguments={},
    )

    with pytest.raises(ValueError, match="available_budget"):
        ToolGateway(create_repository_tool_registry()).execute(
            request,
            context=context,
            available_budget=-1,
            audit_log=ToolAuditLog("run-1"),
        )


def test_gateway_denies_request_bound_to_another_repository(tmp_path: Path) -> None:
    first_repository = tmp_path / "first"
    second_repository = tmp_path / "second"
    first_repository.mkdir()
    second_repository.mkdir()
    first_context = ToolExecutionContext(first_repository)
    second_context = ToolExecutionContext(second_repository)
    request = first_context.bind_call(
        call_id="call-1",
        tool_name="list_tree",
        arguments={},
    )

    result = ToolGateway(create_repository_tool_registry()).execute(
        request,
        context=second_context,
        available_budget=1,
        audit_log=ToolAuditLog("run-1"),
    )

    assert result.decision.reason is ToolDecisionReason.CAPABILITY_DENIED
    assert result.budget_consumed == 0


def test_gateway_normalizes_line_endings_before_hashing(tmp_path: Path) -> None:
    descriptor = _descriptor("normalized_read")
    result, _ = _execute(
        _gateway_for(descriptor, _return_normalized_text),
        ToolExecutionContext(tmp_path),
        tool_name=descriptor.name,
        arguments={},
    )

    assert result.content == "first\nsecond\nthird"
    assert result.audit_record.output_bytes == len(result.content.encode("utf-8"))
    assert result.decision.outcome is ToolDecisionOutcome.ALLOWED


def test_gateway_enforces_utf8_byte_limit_in_isolated_process(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor("oversized_read", max_output_bytes=8)
    result, _ = _execute(
        _gateway_for(descriptor, _return_oversized_text),
        ToolExecutionContext(tmp_path),
        tool_name=descriptor.name,
        arguments={},
    )

    assert result.decision.reason is ToolDecisionReason.OVERSIZED_OUTPUT
    assert result.budget_consumed == 1
    assert result.audit_record.output_bytes is None
    assert result.audit_record.result_digest is None
    assert "界" not in result.content


def test_gateway_terminates_execution_after_descriptor_timeout(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor("slow_read", timeout_ms=25)
    result, _ = _execute(
        _gateway_for(descriptor, _sleep_past_deadline),
        ToolExecutionContext(tmp_path),
        tool_name=descriptor.name,
        arguments={},
    )

    assert result.decision.reason is ToolDecisionReason.TIMEOUT
    assert result.budget_consumed == 1
    assert result.audit_record.latency_ms is not None
    assert result.audit_record.latency_ms >= descriptor.timeout_ms


@pytest.mark.parametrize(
    "executor",
    [_raise_secret_error, _return_invalid_unicode],
    ids=["exception", "invalid-unicode"],
)
def test_gateway_redacts_execution_failures(
    tmp_path: Path,
    executor,
) -> None:
    descriptor = _descriptor("failing_read")
    result, _ = _execute(
        _gateway_for(descriptor, executor),
        ToolExecutionContext(tmp_path),
        tool_name=descriptor.name,
        arguments={},
    )

    assert result.decision.reason is ToolDecisionReason.EXECUTION_ERROR
    assert result.budget_consumed == 1
    assert "super-secret" not in result.content
    assert "authorization" not in result.audit_record.model_dump_json()


def test_audit_log_is_append_only_and_assigns_contiguous_sequences(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    gateway = ToolGateway(create_repository_tool_registry())
    audit_log = ToolAuditLog("run-1")

    first, _ = _execute(
        gateway,
        context,
        tool_name="list_tree",
        arguments={},
        call_id="call-1",
        audit_log=audit_log,
    )
    snapshot = audit_log.records
    second, _ = _execute(
        gateway,
        context,
        tool_name="list_tree",
        arguments={},
        call_id="call-2",
        audit_log=audit_log,
    )

    assert [record.sequence for record in audit_log.records] == [0, 1]
    assert snapshot == (first.audit_record,)
    assert audit_log.records == (first.audit_record, second.audit_record)
    with pytest.raises(TypeError):
        audit_log.records[0] = second.audit_record  # type: ignore[index]


def test_audit_log_rejects_non_contiguous_or_cross_run_history(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    result, _ = _execute(
        ToolGateway(create_repository_tool_registry()),
        context,
        tool_name="list_tree",
        arguments={},
    )
    invalid_sequence = result.audit_record.model_copy(update={"sequence": 2})
    wrong_run = result.audit_record.model_copy(update={"run_id": "run-2"})

    with pytest.raises(ValueError, match="contiguous"):
        ToolAuditLog("run-1", records=(invalid_sequence,))
    with pytest.raises(ValueError, match="run_id"):
        ToolAuditLog("run-1", records=(wrong_run,))
