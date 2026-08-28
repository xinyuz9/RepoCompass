"""Explicit, repository-bound mutation boundary for the local dashboard."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from threading import RLock
from typing import Protocol

from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.dashboard.contracts import (
    DashboardHumanResponseCommand,
    DashboardMemoryAudit,
    DashboardMemoryDeleteCommand,
    DashboardMemoryDeleteResult,
    DashboardThreadDetail,
)
from repocompass.memory.contracts import MemoryAuditReason, MemoryTombstone
from repocompass.service import (
    InvestigationOperationPendingError,
    InvestigationOperationResult,
    InvestigationServiceError,
    ResumeInvestigationRequest,
    ServiceOperationConflictError,
)
from repocompass.tools.gateway import ToolExecutionContext


class DashboardCommandError(RuntimeError):
    """Safe base error for local dashboard mutations."""


class DashboardCommandConflictError(DashboardCommandError):
    """The command does not match current state or prior idempotent identity."""


class DashboardCommandUnavailableError(DashboardCommandError):
    """The selected mutation capability is disabled or temporarily unavailable."""


class DashboardCommandQuery(Protocol):
    @property
    def repository_context(self) -> ToolExecutionContext: ...

    def get_thread(self, thread_id: str) -> DashboardThreadDetail: ...

    def get_memory_audit(self, memory_id: str) -> DashboardMemoryAudit: ...


class DashboardLifecycleWriter(Protocol):
    def replay_resume_investigation(
        self, request: ResumeInvestigationRequest
    ) -> InvestigationOperationResult | None: ...

    def resume_investigation(
        self, request: ResumeInvestigationRequest
    ) -> InvestigationOperationResult: ...


class DashboardMemoryWriter(Protocol):
    def delete_memory(
        self,
        memory_id: str,
        *,
        context: ToolExecutionContext,
        deleted_at_ms: int,
        reason: MemoryAuditReason,
        run_id: str | None = None,
        actor_digest: str | None = None,
    ) -> MemoryTombstone | None: ...


class DashboardCommandService:
    """Admit only interrupt responses and audited memory deletion."""

    def __init__(
        self,
        query_service: DashboardCommandQuery,
        *,
        lifecycle_service: DashboardLifecycleWriter | None = None,
        memory_store: DashboardMemoryWriter | None = None,
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        if not all(
            callable(getattr(query_service, name, None))
            for name in ("get_thread", "get_memory_audit")
        ) or not isinstance(
            getattr(query_service, "repository_context", None),
            ToolExecutionContext,
        ):
            raise TypeError("query_service does not implement the command boundary")
        if lifecycle_service is not None and not all(
            callable(getattr(lifecycle_service, name, None))
            for name in ("replay_resume_investigation", "resume_investigation")
        ):
            raise TypeError("lifecycle_service does not implement lifecycle commands")
        if memory_store is not None and not callable(
            getattr(memory_store, "delete_memory", None)
        ):
            raise TypeError("memory_store does not implement memory deletion")
        if not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self._query_service = query_service
        self._lifecycle_service = lifecycle_service
        self._memory_store = memory_store
        self._clock_ms = clock_ms
        self._mutation_lock = RLock()

    @property
    def lifecycle_mutations_enabled(self) -> bool:
        return self._lifecycle_service is not None

    @property
    def memory_mutations_enabled(self) -> bool:
        return self._memory_store is not None

    def respond_to_review(
        self,
        thread_id: str,
        command: DashboardHumanResponseCommand,
    ) -> InvestigationOperationResult:
        """Consume one exact interrupt, with durable service-level idempotency."""

        if not isinstance(command, DashboardHumanResponseCommand):
            raise TypeError("command must be a DashboardHumanResponseCommand")
        service = self._lifecycle_service
        if service is None:
            raise DashboardCommandUnavailableError(
                "lifecycle mutations are not enabled"
            )
        with self._mutation_lock:
            detail = self._query_service.get_thread(thread_id)
            request = ResumeInvestigationRequest(
                request_id=command.request_id,
                thread_id=detail.thread.thread_id,
                interrupt_id=command.interrupt_id,
                decision=command.decision,
            )
            try:
                replay = service.replay_resume_investigation(request)
                if replay is not None:
                    return replay
                view = detail.thread
                if view.execution_state is not ThreadExecutionState.INTERRUPTED:
                    raise DashboardCommandConflictError(
                        "thread is not awaiting a Human review response"
                    )
                if view.interrupt_id != command.interrupt_id:
                    raise DashboardCommandConflictError(
                        "interrupt_id does not match the pending review"
                    )
                review = view.pending_review
                if (
                    review is None
                    or command.decision.action not in review.allowed_actions
                ):
                    raise DashboardCommandConflictError(
                        "decision is not allowed by the pending review"
                    )
                result = service.resume_investigation(request)
            except DashboardCommandError:
                raise
            except ServiceOperationConflictError as error:
                raise DashboardCommandConflictError(
                    "idempotency key conflicts with a prior command"
                ) from error
            except InvestigationOperationPendingError as error:
                raise DashboardCommandUnavailableError(
                    "a prior lifecycle command is pending recovery"
                ) from error
            except (InvestigationServiceError, ValueError) as error:
                raise DashboardCommandConflictError(
                    "lifecycle command was rejected by the current checkpoint"
                ) from error
            if result.thread_id != detail.thread.thread_id:
                raise DashboardCommandUnavailableError(
                    "lifecycle response identity did not reconcile"
                )
            return result

    def delete_memory(
        self,
        memory_id: str,
        command: DashboardMemoryDeleteCommand,
    ) -> DashboardMemoryDeleteResult:
        """Delete one exact repository-bound memory and preserve its tombstone."""

        if not isinstance(command, DashboardMemoryDeleteCommand):
            raise TypeError("command must be a DashboardMemoryDeleteCommand")
        store = self._memory_store
        if store is None:
            raise DashboardCommandUnavailableError("memory mutations are not enabled")
        if command.confirm_memory_id != memory_id:
            raise DashboardCommandConflictError(
                "memory confirmation does not match the selected identity"
            )
        with self._mutation_lock:
            audit = self._query_service.get_memory_audit(memory_id)
            if audit.tombstone is not None:
                return DashboardMemoryDeleteResult(
                    request_id=command.request_id,
                    memory_id=memory_id,
                    tombstone=audit.tombstone,
                    idempotent_replay=True,
                )
            actor_digest = hashlib.sha256(
                b"repocompass:dashboard-actor:v1\0" + command.request_id.encode("utf-8")
            ).hexdigest()
            tombstone = store.delete_memory(
                memory_id,
                context=self._query_service.repository_context,
                deleted_at_ms=self._clock_ms(),
                reason=MemoryAuditReason.USER_REQUEST,
                actor_digest=actor_digest,
            )
            if tombstone is None:
                raise DashboardCommandUnavailableError(
                    "memory changed before deletion completed"
                )
            return DashboardMemoryDeleteResult(
                request_id=command.request_id,
                memory_id=memory_id,
                tombstone=tombstone,
                idempotent_replay=False,
            )
