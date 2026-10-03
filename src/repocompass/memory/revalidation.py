"""Race-resistant source and snapshot revalidation for recalled memory."""

from repocompass.memory.contracts import (
    MemoryFreshnessStatus,
    MemoryStalenessReason,
    RepositoryMemory,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.source_reader import (
    MAX_SOURCE_BYTES as MAX_MEMORY_SOURCE_BYTES,
    SourceReadError as MemorySourceReadError,
    SourceReadReason as MemorySourceReadReason,
    read_source_bytes as read_memory_source_bytes,
    read_source_digest as read_memory_source_digest,
)
from repocompass.tools.gateway import ToolExecutionContext

# Preserve callers' established memory API without a tools -> memory dependency.
__all__ = [
    "MAX_MEMORY_SOURCE_BYTES",
    "MemorySourceReadError",
    "MemorySourceReadReason",
    "read_memory_source_bytes",
    "read_memory_source_digest",
    "classify_repository_memory_freshness",
    "repository_snapshot_is_clean",
]


def classify_repository_memory_freshness(
    memory: RepositoryMemory,
    context: ToolExecutionContext,
) -> tuple[MemoryFreshnessStatus, MemoryStalenessReason | None]:
    """Classify one recalled fact without ever auto-upgrading an old source."""

    if not isinstance(memory, RepositoryMemory):
        raise TypeError("memory must be a RepositoryMemory")
    if not isinstance(context, ToolExecutionContext):
        raise TypeError("context must be a ToolExecutionContext")
    identity = resolve_memory_repository_identity(context)
    if memory.repository_key != identity.repository_key:
        raise ValueError("memory repository key does not match current repository")
    if context.commit_sha is None:
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.UNVERSIONED_SNAPSHOT,
        )
    if memory.commit_sha != context.commit_sha:
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.COMMIT_MISMATCH,
        )
    if memory.source_repository_id != context.repository_id:
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.SOURCE_REPOSITORY_MISMATCH,
        )

    for source in memory.sources:
        try:
            current_digest = read_memory_source_digest(
                context.repository_root,
                source.location.path,
            )
        except MemorySourceReadError as error:
            if error.reason is MemorySourceReadReason.CHANGED_DURING_READ:
                reason = MemoryStalenessReason.SOURCE_CHANGED
            else:
                reason = MemoryStalenessReason.SOURCE_MISSING
            return MemoryFreshnessStatus.STALE, reason
        if current_digest != source.source_digest:
            return (
                MemoryFreshnessStatus.STALE,
                MemoryStalenessReason.SOURCE_CHANGED,
            )

    if not repository_snapshot_is_clean(context):
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.DIRTY_WORKTREE,
        )
    return MemoryFreshnessStatus.CURRENT, None


def repository_snapshot_is_clean(context: ToolExecutionContext) -> bool:
    """Reuse the Gateway's public immutable-snapshot decision boundary."""

    request = context.bind_call(
        call_id="memory-snapshot-check",
        tool_name="read_code",
        arguments={},
    )
    return context.allows_result_reuse(request)
