"""Repository-bound query service for the local RepoCompass dashboard."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue, ValidationError

from repocompass.agent.cost import PricingTable
from repocompass.agent.conversation import PublicConversationPage
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.runner import ResumeRequest
from repocompass.agent.thread_view import ThreadInvestigationView
from repocompass.agent.trace import TraceEvent
from repocompass.dashboard.contracts import (
    DashboardAvailability,
    DashboardMemoryAudit,
    DashboardMemoryCatalog,
    DashboardRunDetail,
    DashboardRunSummary,
    DashboardSourceLine,
    DashboardSourceSnippet,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    DashboardThreadDetail,
    DashboardThreadSummary,
    DashboardThreadView,
    EvaluatorReplayFrame,
    PlanReplayFrame,
    PublicTraceEvent,
)
from repocompass.dashboard.trace_store import JsonlTraceStore, TraceStoreError
from repocompass.dashboard.events import EventPage
from repocompass.memory.contracts import MemoryScope
from repocompass.memory.store import MemoryStore
from repocompass.service.contracts import InvestigationOperationKind
from repocompass.storage.raw_store import (
    RawStoreConflictError,
    RawStoreError,
    AttemptExecutionMode,
    RawRunAttempt,
    RawRunStore,
)
from repocompass.structure.contracts import MAX_INDEXED_SOURCE_BYTES, SourceLocation
from repocompass.tools.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN
from repocompass.tools.gateway import ToolExecutionContext

_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)
_DIGEST_RE = re.compile(SHA256_PATTERN)
_MAX_CATALOG_SCAN = 1_000
_MAX_THREAD_ATTEMPTS = 100
_MAX_MEMORY_RESULTS = 50
_MAX_SOURCE_SNIPPET_LINES = 160
_SOURCE_CONTEXT_LINES = 4
_DROP_TRACE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "cwd",
        "human_feedback",
        "messages",
        "password",
        "proxy_authorization",
        "reasoning",
        "reasoning_content",
        "refresh_token",
        "repo_path",
        "repository_path",
        "repository_root",
        "secret",
        "token",
        "workspace_root",
    }
)
_DROP_TRACE_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_password",
    "_refresh_token",
    "_secret",
)


class DashboardServiceError(RuntimeError):
    """Safe base error for dashboard adapters."""


class DashboardNotFoundError(DashboardServiceError):
    """A repository-bound run, thread, or memory identity does not exist."""


class DashboardSnapshotConflictError(DashboardServiceError):
    """Persisted evidence no longer matches the visible repository snapshot."""


class DashboardThreadReader(Protocol):
    """Minimal read-only runner boundary required by the dashboard."""

    def inspect_thread(self, request: ResumeRequest) -> ThreadInvestigationView: ...


class DashboardQueryService:
    """Build bounded UI projections without exposing raw artifacts or roots."""

    def __init__(
        self,
        runner: DashboardThreadReader,
        *,
        raw_run_store: RawRunStore,
        trace_store: JsonlTraceStore,
        repository_root: Path,
        memory_store: MemoryStore | None = None,
        pricing_table: PricingTable | None = None,
        lifecycle_mutations_enabled: bool = False,
        memory_mutations_enabled: bool = False,
        cancellation_enabled: bool = False,
        demo_mode: bool = False,
    ) -> None:
        if not callable(getattr(runner, "inspect_thread", None)):
            raise TypeError("runner must implement inspect_thread")
        if not isinstance(raw_run_store, RawRunStore):
            raise TypeError("raw_run_store must be a RawRunStore")
        if not isinstance(trace_store, JsonlTraceStore):
            raise TypeError("trace_store must be a JsonlTraceStore")
        if not isinstance(repository_root, Path):
            raise TypeError("repository_root must be a Path")
        if memory_store is not None and not isinstance(memory_store, MemoryStore):
            raise TypeError("memory_store must be a MemoryStore or None")
        if pricing_table is not None and not isinstance(pricing_table, PricingTable):
            raise TypeError("pricing_table must be a PricingTable or None")
        if type(lifecycle_mutations_enabled) is not bool:
            raise TypeError("lifecycle_mutations_enabled must be a bool")
        if type(memory_mutations_enabled) is not bool:
            raise TypeError("memory_mutations_enabled must be a bool")
        if type(demo_mode) is not bool:
            raise TypeError("demo_mode must be a bool")
        if memory_mutations_enabled and memory_store is None:
            raise ValueError("memory mutations require a memory store")

        self._runner = runner
        self._raw_run_store = raw_run_store
        self._trace_store = trace_store
        self._repository_root = repository_root.expanduser().resolve(strict=True)
        self._context = ToolExecutionContext(self._repository_root)
        self._memory_store = memory_store
        self._pricing_table = pricing_table
        self._lifecycle_mutations_enabled = lifecycle_mutations_enabled
        self._memory_mutations_enabled = memory_mutations_enabled
        if type(cancellation_enabled) is not bool or (
            cancellation_enabled and not lifecycle_mutations_enabled
        ):
            raise ValueError("cancellation requires lifecycle mutations")
        self._cancellation_enabled = cancellation_enabled
        self._demo_mode = demo_mode

    @property
    def repository_context(self) -> ToolExecutionContext:
        """Return a fresh trusted context so HEAD drift is immediately visible."""

        return ToolExecutionContext(self._repository_root)

    def system_info(self) -> DashboardSystemInfo:
        """Describe capabilities without revealing the repository locator."""

        context = self.repository_context
        return DashboardSystemInfo(
            product_version=_product_version(),
            repository_id=context.repository_id,
            commit_sha=context.commit_sha,
            workspace_name=self._repository_root.name or "repository",
            demo_mode=self._demo_mode,
            pricing_configured=self._pricing_table is not None,
            memory_available=self._memory_store is not None,
            lifecycle_mutations_enabled=self._lifecycle_mutations_enabled,
            memory_mutations_enabled=self._memory_mutations_enabled,
            cancellation_enabled=self._cancellation_enabled,
            event_follow_enabled=True,
        )

    def list_threads(self, *, max_results: int = 50) -> DashboardThreadCatalog:
        """List latest repository-bound thread states in stable recency order."""

        _validate_limit(max_results, maximum=100)
        scanned = self._raw_run_store.list_attempts(max_results=_MAX_CATALOG_SCAN)
        attempts = tuple(
            attempt
            for attempt in scanned
            if attempt.repository_id == self._context.repository_id
        )
        grouped: OrderedDict[str, list[RawRunAttempt]] = OrderedDict()
        for attempt in attempts:
            grouped.setdefault(attempt.thread_id, []).append(attempt)

        summaries: list[DashboardThreadSummary] = []
        for thread_id, thread_attempts in tuple(grouped.items())[:max_results]:
            view = self._inspect_optional(thread_id)
            latest = thread_attempts[0]
            summaries.append(
                DashboardThreadSummary(
                    thread_id=thread_id,
                    checkpoint_availability=(
                        DashboardAvailability.AVAILABLE
                        if view is not None
                        else DashboardAvailability.NOT_FOUND
                    ),
                    objective=None if view is None else view.objective,
                    execution_state=None if view is None else view.execution_state,
                    turn_number=None if view is None else view.turn_number,
                    termination_status=(
                        None if view is None else view.termination_status
                    ),
                    latest_attempt=latest,
                    attempt_count=len(thread_attempts),
                )
            )
        return DashboardThreadCatalog(
            threads=tuple(summaries),
            scanned_attempt_count=len(scanned),
            truncated=(len(scanned) == _MAX_CATALOG_SCAN or len(grouped) > max_results),
        )

    def get_thread(self, thread_id: str) -> DashboardThreadDetail:
        """Return checkpoint state plus chronological plan/evaluator replay."""

        normalized_thread_id = _validate_identifier(thread_id, name="thread_id")
        view = self._inspect_optional(normalized_thread_id)
        if view is None:
            raise DashboardNotFoundError("thread was not found")
        attempts = tuple(
            attempt
            for attempt in self._raw_run_store.list_attempts(
                max_results=_MAX_THREAD_ATTEMPTS,
                thread_id=normalized_thread_id,
            )
            if attempt.repository_id == self._context.repository_id
        )
        if not attempts:
            raise DashboardNotFoundError("thread was not found")

        loaded: dict[str, tuple[TraceEvent, ...] | None] = {}
        summaries: list[DashboardRunSummary] = []
        for attempt in attempts:
            events = self._load_trace(attempt.run_id)
            loaded[attempt.run_id] = events
            summaries.append(self._run_summary(attempt, events=events))

        plans: list[PlanReplayFrame] = []
        evaluations: list[EvaluatorReplayFrame] = []
        for attempt in sorted(
            attempts,
            key=lambda item: (item.started_at_ms, item.run_id),
        ):
            events = loaded[attempt.run_id]
            if events is None:
                continue
            attempt_plans, attempt_evaluations = _extract_replay(
                attempt=attempt,
                events=events,
            )
            plans.extend(attempt_plans)
            evaluations.extend(attempt_evaluations)
        return DashboardThreadDetail(
            thread=DashboardThreadView.from_internal(view),
            attempts=tuple(summaries),
            plan_timeline=tuple(plans),
            evaluator_timeline=tuple(evaluations),
        )

    def get_conversation(
        self, thread_id: str, *, before_turn: int | None = None, limit: int = 10
    ) -> PublicConversationPage:
        """Authorize the repository before reading any checkpoint history."""
        normalized = _validate_identifier(thread_id, name="thread_id")
        _validate_limit(limit, maximum=20)
        attempts = self._raw_run_store.list_attempts(
            thread_id=normalized, max_results=1
        )
        if not attempts or attempts[0].repository_id != self._context.repository_id:
            raise DashboardNotFoundError("thread was not found")
        reader = getattr(self._runner, "inspect_conversation", None)
        if not callable(reader):
            raise DashboardServiceError("public conversation reader is unavailable")
        try:
            return reader(
                ResumeRequest(thread_id=normalized),
                before_turn=before_turn,
                limit=limit,
            )
        except (ValueError, TypeError):
            raise DashboardServiceError(
                "public conversation failed integrity validation"
            ) from None

    def get_event_page(
        self, run_id: str, *, after_sequence: int, limit: int
    ) -> EventPage:
        """Project a journal page without loading context packs, model I/O or cost."""
        normalized = _validate_identifier(run_id, name="run_id")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("event page limit is invalid")
        attempt = self._raw_run_store.get_attempt(normalized)
        if attempt is None or attempt.repository_id != self._context.repository_id:
            raise DashboardNotFoundError("run was not found")
        try:
            events = self._raw_run_store.page_trace_events(
                normalized,
                after_sequence=after_sequence,
                limit=limit + 1,
            )
            if events is None:
                legacy = self._trace_store.load(normalized)
                if legacy is not None:
                    if after_sequence >= len(legacy):
                        raise RawStoreConflictError("trace cursor is ahead")
                    events = tuple(e for e in legacy if e.sequence > after_sequence)
        except RawStoreConflictError as error:
            raise DashboardSnapshotConflictError(
                "event cursor does not match persisted trace"
            ) from error
        except (TraceStoreError, RawStoreError) as error:
            raise DashboardServiceError("run events could not be decoded") from error
        return EventPage(
            availability=DashboardAvailability.NOT_PERSISTED
            if events is None
            else DashboardAvailability.AVAILABLE,
            status=attempt.status,
            events=tuple(
                _public_trace_event(e, repository_root=self._context.repository_root)
                for e in (events or ())[:limit]
            ),
            truncated=events is not None and len(events) > limit,
        )

    def get_run(self, run_id: str) -> DashboardRunDetail:
        """Return one attempt's safe trace, context lineage, calls, and cost."""

        normalized_run_id = _validate_identifier(run_id, name="run_id")
        attempt = self._raw_run_store.get_attempt(normalized_run_id)
        if attempt is None or attempt.repository_id != self._context.repository_id:
            raise DashboardNotFoundError("run was not found")
        events = self._load_trace(normalized_run_id)
        if events is None:
            public_events: tuple[PublicTraceEvent, ...] = ()
            plan_timeline: tuple[PlanReplayFrame, ...] = ()
            evaluator_timeline: tuple[EvaluatorReplayFrame, ...] = ()
            trace_availability = DashboardAvailability.NOT_PERSISTED
        else:
            public_events = tuple(
                _public_trace_event(
                    event, repository_root=self._context.repository_root
                )
                for event in events
            )
            plan_timeline, evaluator_timeline = _extract_replay(
                attempt=attempt,
                events=events,
            )
            trace_availability = DashboardAvailability.AVAILABLE

        report = None
        if self._pricing_table is not None:
            report = self._raw_run_store.get_attempt_usage_cost_report(
                normalized_run_id,
                self._pricing_table,
            )
            if report is None:  # pragma: no cover - identity was loaded above
                raise DashboardServiceError("run cost report became unavailable")
        return DashboardRunDetail(
            attempt=attempt,
            trace_availability=trace_availability,
            trace_events=public_events,
            plan_timeline=plan_timeline,
            evaluator_timeline=evaluator_timeline,
            context_packs=self._raw_run_store.list_context_packs(normalized_run_id),
            model_invocations=self._raw_run_store.list_model_invocations(
                normalized_run_id
            ),
            cost_availability=(
                DashboardAvailability.AVAILABLE
                if report is not None
                else DashboardAvailability.NOT_CONFIGURED
            ),
            usage_cost_report=report,
        )

    def get_source(self, thread_id: str, entity_id: str) -> DashboardSourceSnippet:
        """Read one source window only when its persisted digest still matches."""

        normalized_thread_id = _validate_identifier(thread_id, name="thread_id")
        normalized_entity_id = _validate_digest(entity_id, name="entity_id")
        detail = self.get_thread(normalized_thread_id)
        structure = detail.thread.structure
        if structure is None:
            raise DashboardNotFoundError("thread has no structural source evidence")
        current = self.repository_context
        if (
            structure.repository_id != current.repository_id
            or structure.commit_sha != current.commit_sha
        ):
            raise DashboardSnapshotConflictError(
                "repository snapshot changed after evidence collection"
            )
        step = next(
            (
                item
                for item in structure.reading_path.steps
                if item.entity_id == normalized_entity_id
            ),
            None,
        )
        if step is None or not step.source_locations:
            raise DashboardNotFoundError("entity has no traceable source evidence")
        return _read_source_snippet(
            thread_id=normalized_thread_id,
            entity_id=normalized_entity_id,
            symbol=step.canonical_name,
            context=current,
            location=step.source_locations[0],
        )

    def find_lifecycle_attempt(
        self,
        *,
        thread_id: str,
        kind: InvestigationOperationKind,
        reserved_at_ms: int,
        run_id: str | None,
        identity_bound: bool = False,
    ) -> RawRunAttempt | None:
        """Find only the attempt attributable to one durable operation window."""

        normalized_thread_id = _validate_identifier(thread_id, name="thread_id")
        if not isinstance(kind, InvestigationOperationKind):
            raise TypeError("kind must be an InvestigationOperationKind")
        if (
            type(reserved_at_ms) is not int
            or not 0 <= reserved_at_ms <= 9_007_199_254_740_991
        ):
            raise ValueError("reserved_at_ms is invalid")
        expected_mode = AttemptExecutionMode(kind.value)
        if run_id is not None:
            normalized_run_id = _validate_identifier(run_id, name="run_id")
            loaded = self._raw_run_store.get_attempt(normalized_run_id)
            attempts = () if loaded is None else (loaded,)
        else:
            attempts = self._raw_run_store.list_attempts(
                max_results=_MAX_THREAD_ATTEMPTS,
                thread_id=normalized_thread_id,
            )
        return next(
            (
                attempt
                for attempt in attempts
                if attempt.thread_id == normalized_thread_id
                and attempt.repository_id == self._context.repository_id
                and attempt.execution_mode is expected_mode
                and (
                    (identity_bound and run_id is not None)
                    or attempt.started_at_ms >= reserved_at_ms
                )
            ),
            None,
        )

    def list_memory(
        self,
        *,
        max_results: int = 50,
        memory_scope: MemoryScope | None = None,
    ) -> DashboardMemoryCatalog:
        """Return payload-free memory metadata and deletion markers."""

        _validate_limit(max_results, maximum=_MAX_MEMORY_RESULTS)
        if memory_scope is not None and not isinstance(memory_scope, MemoryScope):
            raise TypeError("memory_scope must be a MemoryScope or None")
        if self._memory_store is None:
            return DashboardMemoryCatalog(
                availability=DashboardAvailability.NOT_CONFIGURED,
                entries=(),
                tombstones=(),
            )
        context = self.repository_context
        return DashboardMemoryCatalog(
            availability=DashboardAvailability.AVAILABLE,
            entries=self._memory_store.list_memory_metadata(
                context=context,
                max_results=max_results,
                memory_scope=memory_scope,
            ),
            tombstones=self._memory_store.list_memory_tombstones(
                context=context,
                max_results=max_results,
            ),
        )

    def get_memory_audit(self, memory_id: str) -> DashboardMemoryAudit:
        """Return one payload-free live/deleted memory audit trail."""

        normalized_memory_id = _validate_digest(memory_id, name="memory_id")
        if self._memory_store is None:
            raise DashboardServiceError("memory store is not configured")
        context = self.repository_context
        metadata = self._memory_store.get_memory_metadata(
            normalized_memory_id,
            context=context,
        )
        tombstone = self._memory_store.get_memory_tombstone(
            normalized_memory_id,
            context=context,
        )
        if metadata is None and tombstone is None:
            raise DashboardNotFoundError("memory was not found")
        return DashboardMemoryAudit(
            memory_id=normalized_memory_id,
            metadata=metadata,
            tombstone=tombstone,
            events=self._memory_store.list_memory_audit_events(normalized_memory_id),
        )

    def _inspect_optional(self, thread_id: str) -> ThreadInvestigationView | None:
        try:
            return self._runner.inspect_thread(ResumeRequest(thread_id=thread_id))
        except ValueError as error:
            if str(error).startswith("Checkpoint thread does not exist:"):
                return None
            raise DashboardServiceError(
                "thread checkpoint could not be decoded"
            ) from error
        except (OSError, TypeError, ValidationError) as error:
            raise DashboardServiceError(
                "thread checkpoint could not be read"
            ) from error

    def _load_trace(self, run_id: str) -> tuple[TraceEvent, ...] | None:
        try:
            events = self._raw_run_store.list_trace_events(run_id)
            return events if events is not None else self._trace_store.load(run_id)
        except (TraceStoreError, RawStoreError) as error:
            raise DashboardServiceError("run trace could not be decoded") from error

    def _run_summary(
        self,
        attempt: RawRunAttempt,
        *,
        events: tuple[TraceEvent, ...] | None,
    ) -> DashboardRunSummary:
        return DashboardRunSummary(
            attempt=attempt,
            trace_availability=(
                DashboardAvailability.AVAILABLE
                if events is not None
                else DashboardAvailability.NOT_PERSISTED
            ),
            cost_availability=(
                DashboardAvailability.AVAILABLE
                if self._pricing_table is not None
                else DashboardAvailability.NOT_CONFIGURED
            ),
        )


def _extract_replay(
    *,
    attempt: RawRunAttempt,
    events: Sequence[TraceEvent],
) -> tuple[tuple[PlanReplayFrame, ...], tuple[EvaluatorReplayFrame, ...]]:
    plans: list[PlanReplayFrame] = []
    evaluations: list[EvaluatorReplayFrame] = []
    revision_count = 0
    try:
        for event in events:
            revision_value = event.payload.get("plan_revision_count")
            if revision_value is not None:
                if type(revision_value) is not int or revision_value < 0:
                    raise ValueError("trace plan revision count is invalid")
                revision_count = revision_value
            plan_payload = event.payload.get("investigation_plan")
            if plan_payload is not None:
                plan = InvestigationPlan.model_validate(plan_payload, strict=False)
                plans.append(
                    PlanReplayFrame(
                        run_id=attempt.run_id,
                        turn_number=attempt.turn_number,
                        sequence=event.sequence,
                        node=event.node or "run",
                        revision_count=revision_count,
                        plan=plan,
                    )
                )
            evaluation_payload = event.payload.get("evaluation_report")
            if evaluation_payload is not None:
                if not isinstance(evaluation_payload, Mapping):
                    raise TypeError("trace evaluation report is invalid")
                evaluation_data = dict(evaluation_payload)
                evaluation_data.pop("passed", None)
                report = EvaluationReport.model_validate(
                    evaluation_data,
                    strict=False,
                )
                evaluations.append(
                    EvaluatorReplayFrame(
                        run_id=attempt.run_id,
                        turn_number=attempt.turn_number,
                        sequence=event.sequence,
                        node=event.node or "run",
                        report=report,
                    )
                )
    except (TypeError, ValueError, ValidationError) as error:
        raise DashboardServiceError(
            "run replay artifact could not be decoded"
        ) from error
    return tuple(plans), tuple(evaluations)


def _public_trace_event(
    event: TraceEvent,
    *,
    repository_root: Path,
) -> PublicTraceEvent:
    return PublicTraceEvent(
        run_id=event.run_id,
        sequence=event.sequence,
        kind=event.kind,
        node=event.node,
        payload=_sanitize_mapping(event.payload, repository_root=repository_root),
    )


def _sanitize_mapping(
    value: Mapping[str, JsonValue],
    *,
    repository_root: Path,
) -> dict[str, JsonValue]:
    sanitized: dict[str, JsonValue] = {}
    for key, item in value.items():
        normalized_key = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
        if normalized_key in _DROP_TRACE_KEYS or normalized_key.endswith(
            _DROP_TRACE_SUFFIXES
        ):
            continue
        sanitized[key] = _sanitize_json(item, repository_root=repository_root)
    return sanitized


def _sanitize_json(value: JsonValue, *, repository_root: Path) -> JsonValue:
    if isinstance(value, dict):
        return _sanitize_mapping(value, repository_root=repository_root)
    if isinstance(value, list):
        return [_sanitize_json(item, repository_root=repository_root) for item in value]
    if isinstance(value, str):
        result = value
        root_variants = {
            str(repository_root),
            repository_root.as_posix(),
            str(repository_root).replace("\\", "/"),
        }
        for root in sorted(root_variants, key=len, reverse=True):
            if root:
                result = result.replace(root, "<repository>")
        return result
    return value


def _validate_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _read_source_snippet(
    *,
    thread_id: str,
    entity_id: str,
    symbol: str,
    context: ToolExecutionContext,
    location: SourceLocation,
) -> DashboardSourceSnippet:
    if (
        location.repository_id != context.repository_id
        or location.commit_sha != context.commit_sha
    ):
        raise DashboardSnapshotConflictError(
            "source evidence targets a different repository snapshot"
        )
    relative = Path(location.path)
    current = context.repository_root
    try:
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise DashboardSnapshotConflictError(
                    "source evidence cannot traverse a symbolic link"
                )
        target = current.resolve(strict=True)
        target.relative_to(context.repository_root)
        if not target.is_file() or target.stat().st_size > MAX_INDEXED_SOURCE_BYTES:
            raise DashboardSnapshotConflictError(
                "source evidence file is unavailable or exceeds the display limit"
            )
        encoded = target.read_bytes()
    except DashboardSnapshotConflictError:
        raise
    except (OSError, ValueError) as error:
        raise DashboardSnapshotConflictError(
            "source evidence file is no longer available"
        ) from error
    if hashlib.sha256(encoded).hexdigest() != location.source_digest:
        raise DashboardSnapshotConflictError(
            "source content changed after evidence collection"
        )
    try:
        source_lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise DashboardSnapshotConflictError(
            "source evidence is not valid UTF-8 text"
        ) from error
    if not source_lines or location.line_start > len(source_lines):
        raise DashboardSnapshotConflictError("source evidence range no longer exists")
    if location.line_end > len(source_lines) and not (
        location.line_end == len(source_lines) + 1 and location.column_end == 0
    ):
        raise DashboardSnapshotConflictError("source evidence range no longer exists")
    evidence_line_end = min(location.line_end, len(source_lines))
    snippet_start = max(1, location.line_start - _SOURCE_CONTEXT_LINES)
    desired_end = min(len(source_lines), evidence_line_end + _SOURCE_CONTEXT_LINES)
    snippet_end = min(
        desired_end,
        snippet_start + _MAX_SOURCE_SNIPPET_LINES - 1,
    )
    selected = source_lines[snippet_start - 1 : snippet_end]
    if any(len(line) > 20_000 for line in selected):
        raise DashboardSnapshotConflictError(
            "source evidence contains a line that exceeds the display limit"
        )
    return DashboardSourceSnippet(
        thread_id=thread_id,
        entity_id=entity_id,
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
        path=location.path,
        symbol=symbol,
        source_digest=location.source_digest,
        evidence_line_start=location.line_start,
        evidence_line_end=evidence_line_end,
        snippet_line_start=snippet_start,
        snippet_line_end=snippet_end,
        total_lines=len(source_lines),
        truncated_before=snippet_start > 1,
        truncated_after=snippet_end < len(source_lines),
        lines=tuple(
            DashboardSourceLine(number=number, text=line)
            for number, line in enumerate(selected, start=snippet_start)
        ),
    )


def _validate_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _validate_limit(value: object, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"max_results must be between 1 and {maximum}")
    return value


def _product_version() -> str:
    try:
        return version("repocompass")
    except PackageNotFoundError:  # pragma: no cover - editable installs have metadata
        return "0.1.0"
