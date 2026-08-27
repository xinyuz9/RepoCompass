"""Deterministic, bounded retrieval of memory as non-evidence context."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from repocompass.agent.context import (
    ContextItem,
    ContextMemoryFreshness,
    ContextMemoryMetadata,
    ContextMemoryScope,
    ContextMemoryStalenessReason,
    ContextSection,
    ContextSourceKind,
    MemoryRecallDecision,
    MemoryRecallReport,
    ProvenanceKind,
    ProvenanceLink,
    VerificationStatus,
    create_context_item,
    create_context_source,
    create_memory_recall_report,
)
from repocompass.agent.planning import PlanStep
from repocompass.memory.contracts import (
    EpisodicMemory,
    MemoryFreshnessStatus,
    MemoryVerificationStatus,
)
from repocompass.memory.store import (
    MemoryStore,
    RepositoryMemorySearchResult,
)
from repocompass.tools.gateway import ToolExecutionContext

DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS = 5
DEFAULT_MAX_EPISODIC_MEMORY_RESULTS = 5
MAX_MEMORY_RECALL_RESULTS = 10
_MAX_QUERY_TOKENS = 16
_MAX_QUERY_TOKEN_CHARACTERS = 64
_QUERY_TOKEN_PATTERN = re.compile(r"[\w]+(?:[.:-][\w]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class MemoryRecallResult:
    """Checkpoint-ready memory candidates plus a body-free retrieval report."""

    report: MemoryRecallReport
    context_items: tuple[ContextItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.report, MemoryRecallReport):
            raise TypeError("report must be a MemoryRecallReport")
        if not isinstance(self.context_items, tuple) or not all(
            isinstance(item, ContextItem) for item in self.context_items
        ):
            raise TypeError("context_items must contain ContextItem values")
        item_ids = tuple(item.item_id for item in self.context_items)
        if item_ids != self.report.context_item_ids:
            raise ValueError("memory recall items do not match the report")


def recall_memory_context(
    store: MemoryStore,
    *,
    repo_path: Path,
    objective: str,
    active_step: PlanStep,
    max_repository_results: int = DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
    max_episodic_results: int = DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
    audit_at_ms: int,
    run_id: str | None = None,
) -> MemoryRecallResult:
    """Run exactly one bounded repository and episode search for a new turn."""

    if not isinstance(store, MemoryStore):
        raise TypeError("store must be a MemoryStore")
    if not isinstance(repo_path, Path):
        raise TypeError("repo_path must be a Path")
    if not isinstance(active_step, PlanStep):
        raise TypeError("active_step must be a PlanStep")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be non-empty")
    _validate_limit(max_repository_results, name="max_repository_results")
    _validate_limit(max_episodic_results, name="max_episodic_results")
    if type(audit_at_ms) is not int or audit_at_ms < 0:
        raise ValueError("audit_at_ms must be a non-negative integer")

    context = ToolExecutionContext(repo_path)
    query = build_memory_recall_query(objective, active_step)
    query_digest = _memory_query_digest(query)
    repository_results = store.search_repository_memories(
        query,
        context=context,
        max_results=max_repository_results,
        audit_at_ms=audit_at_ms,
        run_id=run_id,
    )
    episodes = store.search_episodic_memories(
        query,
        context=context,
        max_results=max_episodic_results,
        audit_at_ms=audit_at_ms,
        run_id=run_id,
    )

    decisions: list[MemoryRecallDecision] = []
    context_items: list[ContextItem] = []
    for result in repository_results:
        item = _repository_context_item(result)
        decisions.append(
            MemoryRecallDecision(
                retrieval_id=result.hit.hit_id,
                memory_id=result.memory.memory_id,
                scope=ContextMemoryScope.REPOSITORY,
                freshness=ContextMemoryFreshness(result.hit.freshness.value),
                staleness_reason=(
                    None
                    if result.hit.staleness_reason is None
                    else ContextMemoryStalenessReason(result.hit.staleness_reason.value)
                ),
                context_item_id=None if item is None else item.item_id,
            )
        )
        if item is not None:
            context_items.append(item)

    for rank, episode in enumerate(episodes, start=1):
        retrieval_id = _episode_retrieval_id(
            query_digest=query_digest,
            episode_id=episode.episode_id,
            rank=rank,
        )
        item = _episodic_context_item(
            episode,
            query_digest=query_digest,
            retrieval_id=retrieval_id,
        )
        decisions.append(
            MemoryRecallDecision(
                retrieval_id=retrieval_id,
                memory_id=episode.episode_id,
                scope=ContextMemoryScope.EPISODIC,
                freshness=ContextMemoryFreshness.OBSERVED,
                context_item_id=item.item_id,
            )
        )
        context_items.append(item)

    report = create_memory_recall_report(
        query_digest=query_digest,
        decisions=tuple(decisions),
    )
    return MemoryRecallResult(report=report, context_items=tuple(context_items))


def build_memory_recall_query(objective: str, active_step: PlanStep) -> str:
    """Build a stable FTS-safe query without retaining the source text."""

    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be non-empty")
    if not isinstance(active_step, PlanStep):
        raise TypeError("active_step must be a PlanStep")
    values = (objective, active_step.objective, *active_step.expected_evidence)
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for match in _QUERY_TOKEN_PATTERN.finditer(value.casefold()):
            token = match.group(0)[:_MAX_QUERY_TOKEN_CHARACTERS]
            if not token or token in seen:
                continue
            tokens.append(token)
            seen.add(token)
            if len(tokens) == _MAX_QUERY_TOKENS:
                return " ".join(tokens)
    return " ".join(tokens) if tokens else "repository"


def _repository_context_item(
    result: RepositoryMemorySearchResult,
) -> ContextItem | None:
    memory = result.memory
    hit = result.hit
    if hit.freshness is MemoryFreshnessStatus.STALE:
        return None
    freshness = ContextMemoryFreshness(hit.freshness.value)
    staleness_reason = (
        None
        if hit.staleness_reason is None
        else ContextMemoryStalenessReason(hit.staleness_reason.value)
    )
    metadata = ContextMemoryMetadata(
        memory_id=memory.memory_id,
        retrieval_id=hit.hit_id,
        query_digest=hit.query_digest,
        repository_key=memory.repository_key,
        source_repository_id=memory.source_repository_id,
        source_commit_sha=memory.commit_sha,
        scope=ContextMemoryScope.REPOSITORY,
        freshness=freshness,
        staleness_reason=staleness_reason,
    )
    content = json.dumps(
        {
            "content": memory.content,
            "fact_kind": memory.kind.value,
            "freshness": freshness.value,
            "memory_id": memory.memory_id,
            "source_locations": [
                source.location.model_dump(mode="json") for source in memory.sources
            ],
            "tags": list(memory.tags),
            "title": memory.title,
            "usage_policy": "lead_only_revalidate_with_repository_tools",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if hit.freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED:
        verification = VerificationStatus.UNVERIFIED
        priority = 55
    elif memory.verification_status is MemoryVerificationStatus.HUMAN_APPROVED:
        verification = VerificationStatus.HUMAN_APPROVED
        priority = 72
    else:
        verification = VerificationStatus.INFERRED
        priority = 72
    source = create_context_source(
        kind=ContextSourceKind.MEMORY,
        location=memory.sources[0].location,
        provenance=(ProvenanceLink(kind=ProvenanceKind.MEMORY, id=memory.memory_id),),
        memory=metadata,
    )
    return create_context_item(
        source=source,
        section=ContextSection.MEMORY,
        verification_status=verification,
        content=content,
        estimated_tokens=1,
        priority=priority,
    )


def _episodic_context_item(
    episode: EpisodicMemory,
    *,
    query_digest: str,
    retrieval_id: str,
) -> ContextItem:
    metadata = ContextMemoryMetadata(
        memory_id=episode.episode_id,
        retrieval_id=retrieval_id,
        query_digest=query_digest,
        repository_key=episode.repository_key,
        source_repository_id=episode.source_repository_id,
        source_commit_sha=episode.commit_sha,
        scope=ContextMemoryScope.EPISODIC,
        freshness=ContextMemoryFreshness.OBSERVED,
    )
    content = json.dumps(
        {
            "failed_hypotheses": list(episode.failed_hypotheses),
            "memory_id": episode.episode_id,
            "query_sequence": list(episode.query_sequence),
            "successful_paths": list(episode.successful_paths),
            "summary": episode.summary,
            "termination_status": episode.termination_status,
            "usage_policy": "observed_strategy_history_only_never_code_evidence",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    source = create_context_source(
        kind=ContextSourceKind.MEMORY,
        provenance=(ProvenanceLink(kind=ProvenanceKind.MEMORY, id=episode.episode_id),),
        memory=metadata,
    )
    return create_context_item(
        source=source,
        section=ContextSection.MEMORY,
        verification_status=VerificationStatus.UNVERIFIED,
        content=content,
        estimated_tokens=1,
        priority=40,
    )


def _memory_query_digest(query: str) -> str:
    return hashlib.sha256(
        b"repocompass:memory-query:v1\0" + query.encode("utf-8")
    ).hexdigest()


def _episode_retrieval_id(
    *,
    query_digest: str,
    episode_id: str,
    rank: int,
) -> str:
    payload = json.dumps(
        {
            "episode_id": episode_id,
            "query_digest": query_digest,
            "rank": rank,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(b"repocompass:episodic-memory-hit:v1\0" + payload).hexdigest()


def _validate_limit(value: object, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= MAX_MEMORY_RECALL_RESULTS:
        raise ValueError(f"{name} must be between 1 and {MAX_MEMORY_RECALL_RESULTS}")
