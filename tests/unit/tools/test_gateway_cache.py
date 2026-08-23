"""Cache and duplicate-call tests for the M3.2 Tool Gateway."""

import hashlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import BaseModel

from repocompass.tools.contracts import (
    ToolDecisionOutcome,
    ToolDecisionReason,
    fingerprint_tool_call,
)
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    ToolRegistration,
    ToolRegistry,
    ToolResultCache,
    create_repository_tool_registry,
)
from repocompass.tools.repository_tools import ListTreeArguments


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
        shell=False,
    )


def _clean_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "add", "tracked.py")
    _git(
        repository,
        "-c",
        "user.name=RepoCompass Tests",
        "-c",
        "user.email=repocompass@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    return repository


def _execute(
    gateway: ToolGateway,
    context: ToolExecutionContext,
    audit_log: ToolAuditLog,
    *,
    call_id: str,
    arguments: dict[str, object],
    available_budget: int = 1,
    tool_name: str = "list_tree",
):
    request = context.bind_call(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    return gateway.execute(
        request,
        context=context,
        available_budget=available_budget,
        audit_log=audit_log,
    )


def _sleep_past_deadline(_root: Path, _arguments: BaseModel) -> str:
    time.sleep(0.25)
    return "late"


def _return_text(_root: Path, _arguments: BaseModel) -> str:
    return "stable output"


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"max_entries": 0}, ValueError),
        ({"max_entries": True}, TypeError),
        ({"max_bytes": 0}, ValueError),
        ({"max_bytes": False}, TypeError),
    ],
)
def test_result_cache_requires_positive_integer_bounds(
    overrides: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ToolResultCache(**overrides)


def test_result_cache_validates_content_metadata_and_utf8_byte_bound(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    fingerprint = fingerprint_tool_call(
        context.bind_call(
            call_id="call-cache",
            tool_name="list_tree",
            arguments={},
        )
    )
    cache = ToolResultCache(max_entries=2, max_bytes=6)

    assert cache.put(fingerprint, "界界") is True
    entry = cache.get(fingerprint)
    assert entry is not None
    assert entry.content == "界界"
    assert entry.output_bytes == 6
    assert entry.result_digest == hashlib.sha256("界界".encode()).hexdigest()
    assert cache.entry_count == 1
    assert cache.total_bytes == 6

    second_fingerprint = fingerprint.model_copy(update={"digest": "f" * 64})
    assert cache.put(second_fingerprint, "too large") is False
    assert cache.entry_count == 1
    with pytest.raises(ValueError, match="UTF-8"):
        cache.put(second_fingerprint, "\ud800")


def test_gateway_reuses_equivalent_success_without_execution_budget(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(_clean_repository(tmp_path))
    cache = ToolResultCache(max_entries=8, max_bytes=65_536)
    gateway = ToolGateway(create_repository_tool_registry(), result_cache=cache)
    audit_log = ToolAuditLog("cache-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={"max_depth": 2, "max_entries": 10},
    )
    cached = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-cached",
        arguments={"max_entries": 10, "max_depth": 2},
        available_budget=0,
    )

    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert first.budget_consumed == 1
    assert cached.decision.outcome is ToolDecisionOutcome.CACHED
    assert cached.decision.reason is ToolDecisionReason.CACHE_HIT
    assert cached.budget_consumed == 0
    assert cached.content == first.content
    assert cached.audit_record.output_bytes == first.audit_record.output_bytes
    assert cached.audit_record.result_digest == first.audit_record.result_digest
    assert [record.sequence for record in audit_log.records] == [0, 1]
    assert cache.entry_count == 1


def test_concurrent_equivalent_calls_execute_only_once(tmp_path: Path) -> None:
    context = ToolExecutionContext(_clean_repository(tmp_path))
    gateway = ToolGateway(create_repository_tool_registry())

    def invoke(index: int):
        return _execute(
            gateway,
            context,
            ToolAuditLog(f"concurrent-run-{index}"),
            call_id=f"call-{index}",
            arguments={"max_depth": 2, "max_entries": 10},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(invoke, 1)
        second_future = executor.submit(invoke, 2)
        results = [first_future.result(), second_future.result()]

    assert sorted(result.decision.outcome.value for result in results) == [
        ToolDecisionOutcome.ALLOWED.value,
        ToolDecisionOutcome.CACHED.value,
    ]
    assert sum(result.budget_consumed for result in results) == 1
    assert results[0].content == results[1].content
    assert gateway.result_cache.entry_count == 1


def test_gateway_does_not_cache_mutable_non_git_repository(
    tmp_path: Path,
) -> None:
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    context = ToolExecutionContext(tmp_path)
    cache = ToolResultCache(max_entries=8, max_bytes=65_536)
    gateway = ToolGateway(create_repository_tool_registry(), result_cache=cache)
    audit_log = ToolAuditLog("mutable-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={},
    )
    second = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-second",
        arguments={},
    )

    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert first.budget_consumed == second.budget_consumed == 1
    assert cache.entry_count == 0


def test_gateway_bypasses_stale_entry_when_worktree_becomes_dirty(
    tmp_path: Path,
) -> None:
    repository = _clean_repository(tmp_path)
    context = ToolExecutionContext(repository)
    cache = ToolResultCache(max_entries=8, max_bytes=65_536)
    gateway = ToolGateway(create_repository_tool_registry(), result_cache=cache)
    audit_log = ToolAuditLog("dirty-run")
    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={},
    )
    (repository / "tracked.py").write_text("value = 2\n", encoding="utf-8")

    second = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-second",
        arguments={},
    )

    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.budget_consumed == 1


def test_gateway_bypasses_cache_when_ignored_content_exists(
    tmp_path: Path,
) -> None:
    repository = _clean_repository(tmp_path)
    (repository / ".gitignore").write_text("*.generated\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(
        repository,
        "-c",
        "user.name=RepoCompass Tests",
        "-c",
        "user.email=repocompass@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "ignore generated files",
    )
    (repository / "runtime.generated").write_text("value = 1\n", encoding="utf-8")
    context = ToolExecutionContext(repository)
    cache = ToolResultCache(max_entries=8, max_bytes=65_536)
    gateway = ToolGateway(create_repository_tool_registry(), result_cache=cache)
    audit_log = ToolAuditLog("ignored-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={},
    )
    second = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-second",
        arguments={},
    )

    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert first.budget_consumed == second.budget_consumed == 1
    assert cache.entry_count == 0


def test_gateway_cache_isolated_by_commit_identity(tmp_path: Path) -> None:
    repository = _clean_repository(tmp_path)
    cache = ToolResultCache(max_entries=8, max_bytes=65_536)
    gateway = ToolGateway(create_repository_tool_registry(), result_cache=cache)
    first_context = ToolExecutionContext(repository)
    first = _execute(
        gateway,
        first_context,
        ToolAuditLog("first-commit-run"),
        call_id="call-first",
        arguments={},
    )
    (repository / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    _git(repository, "add", "tracked.py")
    _git(
        repository,
        "-c",
        "user.name=RepoCompass Tests",
        "-c",
        "user.email=repocompass@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "second fixture",
    )
    second_context = ToolExecutionContext(repository)

    second = _execute(
        gateway,
        second_context,
        ToolAuditLog("second-commit-run"),
        call_id="call-second",
        arguments={},
        available_budget=1,
    )

    assert first_context.commit_sha != second_context.commit_sha
    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.budget_consumed == 1
    assert cache.entry_count == 2


def test_result_cache_uses_deterministic_lru_eviction(tmp_path: Path) -> None:
    context = ToolExecutionContext(_clean_repository(tmp_path))
    cache = ToolResultCache(max_entries=2, max_bytes=65_536)
    gateway = ToolGateway(create_repository_tool_registry(), result_cache=cache)
    audit_log = ToolAuditLog("lru-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-1",
        arguments={"max_depth": 1},
    )
    second = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-2",
        arguments={"max_depth": 2},
    )
    first_hit = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-1-hit",
        arguments={"max_depth": 1},
    )
    third = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-3",
        arguments={"max_depth": 3},
    )
    evicted_second = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-2-again",
        arguments={"max_depth": 2},
    )

    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert first_hit.decision.outcome is ToolDecisionOutcome.CACHED
    assert third.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert evicted_second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert cache.entry_count == 2


def test_non_cacheable_descriptor_always_executes(tmp_path: Path) -> None:
    repository = _clean_repository(tmp_path)
    descriptor = create_repository_tool_registry().require("list_tree").descriptor
    registration = ToolRegistration(
        descriptor=descriptor.model_copy(update={"cacheable": False}),
        arguments_model=ListTreeArguments,
        executor=_return_text,
    )
    cache = ToolResultCache(max_entries=8, max_bytes=65_536)
    gateway = ToolGateway(ToolRegistry((registration,)), result_cache=cache)
    context = ToolExecutionContext(repository)
    audit_log = ToolAuditLog("non-cacheable-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={},
    )
    second = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-second",
        arguments={},
    )

    assert first.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert second.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert cache.entry_count == 0


def test_repeated_invalid_request_is_deduplicated_after_policy_recheck(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(_clean_repository(tmp_path))
    gateway = ToolGateway(create_repository_tool_registry())
    audit_log = ToolAuditLog("invalid-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={"max_depth": "invalid"},
    )
    duplicate = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-duplicate",
        arguments={"max_depth": "invalid"},
    )

    assert first.decision.outcome is ToolDecisionOutcome.DENIED
    assert first.decision.reason is ToolDecisionReason.INVALID_ARGUMENTS
    assert duplicate.decision.outcome is ToolDecisionOutcome.DEDUPLICATED
    assert duplicate.decision.reason is ToolDecisionReason.DUPLICATE_CALL
    assert duplicate.budget_consumed == 0
    assert duplicate.succeeded is False
    assert "invalid" not in duplicate.content


def test_repeated_execution_failure_is_blocked_before_second_execution(
    tmp_path: Path,
) -> None:
    repository = _clean_repository(tmp_path)
    descriptor = create_repository_tool_registry().require("list_tree").descriptor
    registration = ToolRegistration(
        descriptor=descriptor.model_copy(update={"timeout_ms": 25}),
        arguments_model=ListTreeArguments,
        executor=_sleep_past_deadline,
    )
    gateway = ToolGateway(ToolRegistry((registration,)))
    context = ToolExecutionContext(repository)
    audit_log = ToolAuditLog("timeout-run")

    first = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-first",
        arguments={},
    )
    duplicate = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-duplicate",
        arguments={},
    )

    assert first.decision.reason is ToolDecisionReason.TIMEOUT
    assert first.budget_consumed == 1
    assert duplicate.decision.outcome is ToolDecisionOutcome.DEDUPLICATED
    assert duplicate.decision.reason is ToolDecisionReason.DUPLICATE_CALL
    assert duplicate.budget_consumed == 0
    assert gateway.result_cache.entry_count == 0


def test_deduplication_floor_allows_one_retry_in_a_new_active_step(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(_clean_repository(tmp_path))
    gateway = ToolGateway(create_repository_tool_registry())
    first_log = ToolAuditLog("step-run")
    first = _execute(
        gateway,
        context,
        first_log,
        call_id="call-step-1",
        arguments={"max_depth": "invalid"},
    )
    second_step_log = ToolAuditLog(
        "step-run",
        records=first_log.records,
        deduplication_floor=1,
    )

    retried = _execute(
        gateway,
        context,
        second_step_log,
        call_id="call-step-2",
        arguments={"max_depth": "invalid"},
    )
    duplicate = _execute(
        gateway,
        context,
        second_step_log,
        call_id="call-step-2-duplicate",
        arguments={"max_depth": "invalid"},
    )

    assert first.decision.reason is ToolDecisionReason.INVALID_ARGUMENTS
    assert retried.decision.outcome is ToolDecisionOutcome.DENIED
    assert retried.decision.reason is ToolDecisionReason.INVALID_ARGUMENTS
    assert duplicate.decision.outcome is ToolDecisionOutcome.DEDUPLICATED
    assert [record.sequence for record in second_step_log.records] == [0, 1, 2]


def test_audit_log_rejects_invalid_deduplication_floor() -> None:
    with pytest.raises(TypeError, match="deduplication_floor"):
        ToolAuditLog("run-1", deduplication_floor=True)
    with pytest.raises(ValueError, match="deduplication_floor"):
        ToolAuditLog("run-1", deduplication_floor=1)


def test_budget_denial_does_not_poison_later_execution(tmp_path: Path) -> None:
    context = ToolExecutionContext(_clean_repository(tmp_path))
    gateway = ToolGateway(create_repository_tool_registry())
    audit_log = ToolAuditLog("budget-run")

    denied = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-denied",
        arguments={},
        available_budget=0,
    )
    allowed = _execute(
        gateway,
        context,
        audit_log,
        call_id="call-allowed",
        arguments={},
        available_budget=1,
    )

    assert denied.decision.reason is ToolDecisionReason.BUDGET_EXHAUSTED
    assert allowed.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert allowed.budget_consumed == 1
