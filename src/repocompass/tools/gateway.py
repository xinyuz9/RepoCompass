"""Fail-closed execution boundary for model-requested repository tools."""

import hashlib
import json
import multiprocessing
import os
import re
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ValidationError

from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    ToolAuditRecord,
    ToolCallFingerprint,
    ToolCallRequest,
    ToolCapability,
    ToolDecisionOutcome,
    ToolDecisionReason,
    ToolDescriptor,
    ToolPolicyDecision,
    ToolSideEffect,
    fingerprint_tool_call,
)
from repocompass.tools.repository_tools import (
    REPOSITORY_TOOL_DESCRIPTORS,
    ListTreeArguments,
    ReadCodeArguments,
    SearchTextArguments,
    execute_list_tree,
    execute_read_code,
    execute_search_text,
)

ToolExecutor = Callable[[Path, BaseModel], str]


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """Trusted binding between policy metadata, arguments, and implementation."""

    descriptor: ToolDescriptor
    arguments_model: type[BaseModel]
    executor: ToolExecutor
    repository_path_field: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ToolDescriptor):
            raise TypeError("descriptor must be a ToolDescriptor")
        if not isinstance(self.arguments_model, type) or not issubclass(
            self.arguments_model,
            BaseModel,
        ):
            raise TypeError("arguments_model must be a Pydantic model type")
        if not callable(self.executor):
            raise TypeError("executor must be callable")
        if (
            self.repository_path_field is not None
            and self.repository_path_field not in self.arguments_model.model_fields
        ):
            raise ValueError("repository_path_field is not in the arguments model")


class ToolRegistry:
    """Immutable trusted registry used for deterministic tool lookup."""

    def __init__(self, registrations: Iterable[ToolRegistration]) -> None:
        indexed: dict[str, ToolRegistration] = {}
        for registration in registrations:
            if not isinstance(registration, ToolRegistration):
                raise TypeError("registrations must contain ToolRegistration values")
            name = registration.descriptor.name
            if name in indexed:
                raise ValueError(f"duplicate tool registration: {name}")
            indexed[name] = registration

        if not indexed:
            raise ValueError("tool registry must not be empty")

        self._registrations = MappingProxyType(indexed)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._registrations)

    def get(self, name: str) -> ToolRegistration | None:
        return self._registrations.get(name)

    def require(self, name: str) -> ToolRegistration:
        registration = self.get(name)
        if registration is None:
            raise KeyError(name)
        return registration


@dataclass(frozen=True, slots=True, init=False)
class ToolExecutionContext:
    """Trusted repository identity injected outside model-controlled arguments."""

    repository_root: Path
    repository_id: str
    commit_sha: str | None
    _root_device: int = field(repr=False)
    _root_inode: int = field(repr=False)

    def __init__(self, repository_path: Path) -> None:
        if not isinstance(repository_path, Path):
            raise TypeError("repository_path must be a Path")
        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {repository_path}"
            )
        if not repository_path.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {repository_path}"
            )

        repository_root = repository_path.expanduser().resolve(strict=True)
        root_stat = repository_root.stat()
        repository_id = _repository_identity(repository_root)
        commit_sha = _discover_commit_sha(repository_root)

        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "repository_id", repository_id)
        object.__setattr__(self, "commit_sha", commit_sha)
        object.__setattr__(self, "_root_device", root_stat.st_dev)
        object.__setattr__(self, "_root_inode", root_stat.st_ino)

    def bind_call(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolCallRequest:
        """Bind untrusted model arguments to this trusted repository identity."""

        return ToolCallRequest.model_validate(
            {
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "repository_id": self.repository_id,
                "commit_sha": self.commit_sha,
            }
        )

    def matches(self, request: ToolCallRequest) -> bool:
        """Verify identity binding and detect repository-root replacement."""

        if (
            request.repository_id != self.repository_id
            or request.commit_sha != self.commit_sha
        ):
            return False
        try:
            current_stat = self.repository_root.stat()
        except OSError:
            return False
        return (
            current_stat.st_dev == self._root_device
            and current_stat.st_ino == self._root_inode
            and self.repository_root.is_dir()
            and _discover_commit_sha(self.repository_root) == self.commit_sha
        )


class ToolAuditLog:
    """In-memory append-only audit sequence for one execution attempt."""

    def __init__(
        self,
        run_id: str,
        *,
        records: tuple[ToolAuditRecord, ...] = (),
    ) -> None:
        if (
            not isinstance(run_id, str)
            or re.fullmatch(
                IDENTIFIER_PATTERN,
                run_id,
            )
            is None
        ):
            raise ValueError("run_id is not a canonical audit identifier")
        if not isinstance(records, tuple):
            raise TypeError("records must be a tuple")
        for sequence, record in enumerate(records):
            if not isinstance(record, ToolAuditRecord):
                raise TypeError("records must contain ToolAuditRecord values")
            if record.run_id != run_id:
                raise ValueError("audit record run_id does not match the log")
            if record.sequence != sequence:
                raise ValueError("audit record sequence must be contiguous")

        self._run_id = run_id
        self._records = list(records)
        self._lock = threading.Lock()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def records(self) -> tuple[ToolAuditRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def append(
        self,
        *,
        request: ToolCallRequest,
        fingerprint: ToolCallFingerprint,
        decision: ToolPolicyDecision,
        latency_ms: int,
        output_bytes: int | None,
        result_digest: str | None,
    ) -> ToolAuditRecord:
        """Append exactly one immutable record with a log-owned sequence."""

        with self._lock:
            record = ToolAuditRecord(
                sequence=len(self._records),
                run_id=self._run_id,
                call_id=request.call_id,
                tool_name=request.tool_name,
                repository_id=request.repository_id,
                commit_sha=request.commit_sha,
                fingerprint=fingerprint,
                decision=decision,
                latency_ms=latency_ms,
                output_bytes=output_bytes,
                result_digest=result_digest,
            )
            self._records.append(record)
            return record


@dataclass(frozen=True, slots=True)
class ToolGatewayResult:
    """Bounded result returned to orchestration after one gateway decision."""

    request: ToolCallRequest
    decision: ToolPolicyDecision
    content: str
    budget_consumed: int
    audit_record: ToolAuditRecord

    @property
    def succeeded(self) -> bool:
        return self.decision.outcome in {
            ToolDecisionOutcome.ALLOWED,
            ToolDecisionOutcome.CACHED,
            ToolDecisionOutcome.DEDUPLICATED,
        }


@dataclass(frozen=True, slots=True)
class _IsolatedExecutionResult:
    reason: ToolDecisionReason | None
    output: str | None


class ToolGateway:
    """Validate, authorize, execute, normalize, and audit one tool request."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_capabilities: frozenset[ToolCapability] = frozenset(
            {ToolCapability.FILESYSTEM_READ}
        ),
        allowed_side_effects: frozenset[ToolSideEffect] = frozenset(
            {ToolSideEffect.READ_ONLY}
        ),
        process_context: BaseContext | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        if not isinstance(allowed_capabilities, frozenset) or any(
            not isinstance(capability, ToolCapability)
            for capability in allowed_capabilities
        ):
            raise TypeError("allowed_capabilities must be ToolCapability values")
        if not isinstance(allowed_side_effects, frozenset) or any(
            not isinstance(side_effect, ToolSideEffect)
            for side_effect in allowed_side_effects
        ):
            raise TypeError("allowed_side_effects must be ToolSideEffect values")

        self._registry = registry
        self._allowed_capabilities = allowed_capabilities
        self._allowed_side_effects = allowed_side_effects
        self._process_context = process_context or _default_process_context()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def execute(
        self,
        request: ToolCallRequest,
        *,
        context: ToolExecutionContext,
        available_budget: int,
        audit_log: ToolAuditLog,
    ) -> ToolGatewayResult:
        """Execute one request through the complete fail-closed boundary."""

        if not isinstance(request, ToolCallRequest):
            raise TypeError("request must be a ToolCallRequest")
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("context must be a ToolExecutionContext")
        if not isinstance(audit_log, ToolAuditLog):
            raise TypeError("audit_log must be a ToolAuditLog")
        if (
            not isinstance(available_budget, int)
            or isinstance(available_budget, bool)
            or available_budget < 0
        ):
            raise ValueError("available_budget must be a non-negative integer")

        started_ns = time.perf_counter_ns()
        fingerprint = fingerprint_tool_call(request)

        if not context.matches(request):
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=ToolDecisionReason.CAPABILITY_DENIED,
                started_ns=started_ns,
                audit_log=audit_log,
            )

        registration = self._registry.get(request.tool_name)
        if registration is None:
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=ToolDecisionReason.UNKNOWN_TOOL,
                started_ns=started_ns,
                audit_log=audit_log,
            )

        try:
            parsed_arguments = _parse_arguments(registration, request)
        except (TypeError, ValueError, ValidationError):
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=ToolDecisionReason.INVALID_ARGUMENTS,
                started_ns=started_ns,
                audit_log=audit_log,
            )

        path_reason = _repository_path_policy(
            registration,
            parsed_arguments,
            context.repository_root,
        )
        if path_reason is not None:
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=path_reason,
                started_ns=started_ns,
                audit_log=audit_log,
            )

        descriptor = registration.descriptor
        if (
            not set(descriptor.capabilities).issubset(self._allowed_capabilities)
            or descriptor.side_effect not in self._allowed_side_effects
        ):
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=ToolDecisionReason.CAPABILITY_DENIED,
                started_ns=started_ns,
                audit_log=audit_log,
            )

        if available_budget == 0:
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=ToolDecisionReason.BUDGET_EXHAUSTED,
                started_ns=started_ns,
                audit_log=audit_log,
            )

        isolated_result = _execute_isolated(
            registration,
            parsed_arguments,
            context.repository_root,
            process_context=self._process_context,
        )
        if isolated_result.reason is not None:
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=isolated_result.reason,
                started_ns=started_ns,
                audit_log=audit_log,
                budget_consumed=1,
            )

        output = isolated_result.output
        if output is None:
            return self._deny(
                request,
                fingerprint=fingerprint,
                reason=ToolDecisionReason.EXECUTION_ERROR,
                started_ns=started_ns,
                audit_log=audit_log,
                budget_consumed=1,
            )

        encoded_output = output.encode("utf-8")
        result_digest = hashlib.sha256(encoded_output).hexdigest()
        decision = ToolPolicyDecision(
            outcome=ToolDecisionOutcome.ALLOWED,
            reason=ToolDecisionReason.ALLOWED,
        )
        return self._finish(
            request,
            fingerprint=fingerprint,
            decision=decision,
            content=output,
            budget_consumed=1,
            started_ns=started_ns,
            audit_log=audit_log,
            output_bytes=len(encoded_output),
            result_digest=result_digest,
        )

    def _deny(
        self,
        request: ToolCallRequest,
        *,
        fingerprint: ToolCallFingerprint,
        reason: ToolDecisionReason,
        started_ns: int,
        audit_log: ToolAuditLog,
        budget_consumed: int = 0,
    ) -> ToolGatewayResult:
        decision = ToolPolicyDecision(
            outcome=ToolDecisionOutcome.DENIED,
            reason=reason,
        )
        return self._finish(
            request,
            fingerprint=fingerprint,
            decision=decision,
            content=_safe_failure_content(reason),
            budget_consumed=budget_consumed,
            started_ns=started_ns,
            audit_log=audit_log,
            output_bytes=None,
            result_digest=None,
        )

    @staticmethod
    def _finish(
        request: ToolCallRequest,
        *,
        fingerprint: ToolCallFingerprint,
        decision: ToolPolicyDecision,
        content: str,
        budget_consumed: int,
        started_ns: int,
        audit_log: ToolAuditLog,
        output_bytes: int | None,
        result_digest: str | None,
    ) -> ToolGatewayResult:
        elapsed_ns = max(0, time.perf_counter_ns() - started_ns)
        latency_ms = (elapsed_ns + 999_999) // 1_000_000
        audit_record = audit_log.append(
            request=request,
            fingerprint=fingerprint,
            decision=decision,
            latency_ms=latency_ms,
            output_bytes=output_bytes,
            result_digest=result_digest,
        )
        return ToolGatewayResult(
            request=request,
            decision=decision,
            content=content,
            budget_consumed=budget_consumed,
            audit_record=audit_record,
        )


def create_repository_tool_registry() -> ToolRegistry:
    """Build the immutable registry for the three current repository tools."""

    return ToolRegistry(
        (
            ToolRegistration(
                descriptor=REPOSITORY_TOOL_DESCRIPTORS["list_tree"],
                arguments_model=ListTreeArguments,
                executor=_execute_list_tree,
            ),
            ToolRegistration(
                descriptor=REPOSITORY_TOOL_DESCRIPTORS["search_text"],
                arguments_model=SearchTextArguments,
                executor=_execute_search_text,
            ),
            ToolRegistration(
                descriptor=REPOSITORY_TOOL_DESCRIPTORS["read_code"],
                arguments_model=ReadCodeArguments,
                executor=_execute_read_code,
                repository_path_field="relative_path",
            ),
        )
    )


def _execute_list_tree(repository_root: Path, arguments: BaseModel) -> str:
    if not isinstance(arguments, ListTreeArguments):
        raise TypeError("list_tree received invalid parsed arguments")
    return execute_list_tree(repository_root, arguments)


def _execute_search_text(repository_root: Path, arguments: BaseModel) -> str:
    if not isinstance(arguments, SearchTextArguments):
        raise TypeError("search_text received invalid parsed arguments")
    return execute_search_text(repository_root, arguments)


def _execute_read_code(repository_root: Path, arguments: BaseModel) -> str:
    if not isinstance(arguments, ReadCodeArguments):
        raise TypeError("read_code received invalid parsed arguments")
    return execute_read_code(repository_root, arguments)


def _parse_arguments(
    registration: ToolRegistration,
    request: ToolCallRequest,
) -> BaseModel:
    encoded = json.dumps(
        request.arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return registration.arguments_model.model_validate_json(encoded, strict=True)


def _repository_path_policy(
    registration: ToolRegistration,
    arguments: BaseModel,
    repository_root: Path,
) -> ToolDecisionReason | None:
    field_name = registration.repository_path_field
    if field_name is None:
        return None

    path_text = getattr(arguments, field_name)
    if not isinstance(path_text, str):
        return ToolDecisionReason.INVALID_ARGUMENTS
    relative_path = Path(path_text)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return ToolDecisionReason.PATH_ESCAPE

    candidate = repository_root / relative_path
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(repository_root)
    except ValueError:
        return ToolDecisionReason.PATH_ESCAPE

    current = repository_root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            return ToolDecisionReason.CAPABILITY_DENIED

    try:
        candidate_stat = candidate.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        return ToolDecisionReason.CAPABILITY_DENIED
    if not stat.S_ISREG(candidate_stat.st_mode):
        return ToolDecisionReason.CAPABILITY_DENIED

    return None


def _execute_isolated(
    registration: ToolRegistration,
    arguments: BaseModel,
    repository_root: Path,
    *,
    process_context: BaseContext,
) -> _IsolatedExecutionResult:
    receive_connection, send_connection = process_context.Pipe(duplex=False)
    process = process_context.Process(
        target=_tool_process_worker,
        args=(
            send_connection,
            registration.executor,
            repository_root,
            arguments,
        ),
        daemon=True,
    )
    try:
        process.start()
    except (OSError, RuntimeError, AssertionError):
        receive_connection.close()
        send_connection.close()
        return _IsolatedExecutionResult(
            reason=ToolDecisionReason.EXECUTION_ERROR,
            output=None,
        )
    finally:
        send_connection.close()

    timeout_seconds = registration.descriptor.timeout_ms / 1_000
    try:
        if not receive_connection.poll(timeout_seconds):
            return _IsolatedExecutionResult(
                reason=ToolDecisionReason.TIMEOUT,
                output=None,
            )
        try:
            payload = receive_connection.recv_bytes(
                maxlength=registration.descriptor.max_output_bytes + 1
            )
        except OSError:
            return _IsolatedExecutionResult(
                reason=ToolDecisionReason.OVERSIZED_OUTPUT,
                output=None,
            )
        except EOFError:
            return _IsolatedExecutionResult(
                reason=ToolDecisionReason.EXECUTION_ERROR,
                output=None,
            )

        status_byte = payload[:1]
        if status_byte != b"S":
            return _IsolatedExecutionResult(
                reason=ToolDecisionReason.EXECUTION_ERROR,
                output=None,
            )
        encoded_output = payload[1:]
        if len(encoded_output) > registration.descriptor.max_output_bytes:
            return _IsolatedExecutionResult(
                reason=ToolDecisionReason.OVERSIZED_OUTPUT,
                output=None,
            )
        try:
            output = encoded_output.decode("utf-8")
        except UnicodeDecodeError:
            return _IsolatedExecutionResult(
                reason=ToolDecisionReason.EXECUTION_ERROR,
                output=None,
            )
        return _IsolatedExecutionResult(reason=None, output=output)
    finally:
        receive_connection.close()
        _terminate_process(process)


def _tool_process_worker(
    connection: Connection,
    executor: ToolExecutor,
    repository_root: Path,
    arguments: BaseModel,
) -> None:
    try:
        output = executor(repository_root, arguments)
        if not isinstance(output, str):
            connection.send_bytes(b"E")
            return
        normalized_output = output.replace("\r\n", "\n").replace("\r", "\n")
        connection.send_bytes(b"S" + normalized_output.encode("utf-8"))
    except BaseException:
        try:
            connection.send_bytes(b"E")
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _terminate_process(process: multiprocessing.Process) -> None:
    process.join(timeout=0.05)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.2)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=0.2)
    process.close()


def _safe_failure_content(reason: ToolDecisionReason) -> str:
    return json.dumps(
        {"error": {"reason": reason.value}},
        sort_keys=True,
        separators=(",", ":"),
    )


def _repository_identity(repository_root: Path) -> str:
    encoded_path = os.fsencode(str(repository_root))
    return hashlib.sha256(b"repocompass:repository:v1\0" + encoded_path).hexdigest()


def _discover_commit_sha(repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None

    commit_sha = completed.stdout.strip().lower()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit_sha) is None:
        return None
    return commit_sha


def _default_process_context() -> BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")
