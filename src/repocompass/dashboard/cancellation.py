"""Repository-authorized cancellation, independent of serialized lifecycle locks."""

from pathlib import Path

from repocompass.dashboard.commands import DashboardCommandConflictError
from repocompass.dashboard.service import DashboardNotFoundError
from repocompass.storage.cancellation import (
    CancellationConflict,
    CancellationError,
    CancellationRecord,
    CancellationRequest,
    CancellationStore,
)
from repocompass.storage.raw_store import AttemptStatus, RawRunStore
from repocompass.tools.gateway import ToolExecutionContext


class DashboardCancellationService:
    def __init__(
        self, store: CancellationStore, raw_store: RawRunStore, repository_root: Path
    ):
        self._store = store
        self._raw_store = raw_store
        self._repository_id = ToolExecutionContext(repository_root).repository_id

    def get(self, run_id: str) -> CancellationRecord:
        attempt = self._raw_store.get_attempt(run_id)
        if attempt is None or attempt.repository_id != self._repository_id:
            raise DashboardNotFoundError("run was not found")
        record = self._store.get(run_id)
        if record is None:
            raise DashboardNotFoundError("run is not enrolled for cancellation")
        if any(
            getattr(record, key) != getattr(attempt, key)
            for key in (
                "run_id",
                "thread_id",
                "repository_id",
                "commit_sha",
            )
        ):
            raise CancellationError("cancellation enrollment does not match attempt")
        return record

    def request(self, run_id: str, command: CancellationRequest) -> CancellationRecord:
        if command.run_id != run_id:
            raise DashboardCommandConflictError(
                "cancellation run ID does not match route"
            )
        record = self.get(run_id)
        attempt = self._raw_store.get_attempt(run_id)
        if (
            attempt.status is not AttemptStatus.RUNNING
            and record.request_id != command.request_id
        ):
            raise DashboardCommandConflictError("run is no longer running")
        try:
            return self._store.request(command)
        except CancellationConflict:
            raise DashboardCommandConflictError(
                "cancellation identity conflicts or run already finished"
            ) from None
