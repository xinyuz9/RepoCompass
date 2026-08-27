"""Regenerate the bundled deterministic memory on/off ablation case."""

import argparse
import hashlib
from pathlib import Path

from repocompass.agent.context import (
    ContextMemoryFreshness,
    ContextMemoryMetadata,
    ContextMemoryScope,
    ContextMemoryStalenessReason,
    ContextSection,
    ContextSourceKind,
    MemoryRecallDecision,
    ProvenanceKind,
    ProvenanceLink,
    VerificationStatus,
    create_context_item,
    create_context_source,
    create_memory_recall_report,
)
from repocompass.evaluation.case_file import load_context_ablation_case_file
from repocompass.evaluation.memory_ablation import create_memory_ablation_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_CASE = PROJECT_ROOT / "benchmarks/cases/context-pressure-v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks/cases/memory-recall-v1.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def build_case(context_case_path: Path = DEFAULT_CONTEXT_CASE):
    """Build the case from stable production ContextItem factories."""

    context_case = load_context_ablation_case_file(context_case_path)
    required_ids = set(context_case.required_evidence_item_ids)
    base_candidates = tuple(
        item
        for item in context_case.candidates
        if item.item_id in required_ids or item.pinned
    )
    prior_repository_item = next(
        item
        for item in context_case.candidates
        if item.source.kind is ContextSourceKind.MEMORY
    )
    repository_memory = create_context_item(
        source=prior_repository_item.source,
        section=ContextSection.MEMORY,
        verification_status=VerificationStatus.UNVERIFIED,
        content=prior_repository_item.content,
        estimated_tokens=1,
        priority=55,
    )
    query_digest = repository_memory.source.memory.query_digest  # type: ignore[union-attr]

    episode_metadata = ContextMemoryMetadata(
        memory_id=_digest("memory-ablation-episode"),
        retrieval_id=_digest("memory-ablation-episode-retrieval"),
        query_digest=query_digest,
        repository_key="d" * 64,
        source_repository_id=(
            repository_memory.source.memory.source_repository_id  # type: ignore[union-attr]
        ),
        source_commit_sha="1" * 40,
        scope=ContextMemoryScope.EPISODIC,
        freshness=ContextMemoryFreshness.OBSERVED,
    )
    episode_source = create_context_source(
        kind=ContextSourceKind.MEMORY,
        provenance=(
            ProvenanceLink(
                kind=ProvenanceKind.MEMORY,
                id=episode_metadata.memory_id,
            ),
        ),
        memory=episode_metadata,
    )
    episode_memory = create_context_item(
        source=episode_source,
        section=ContextSection.MEMORY,
        verification_status=VerificationStatus.UNVERIFIED,
        content=(
            '{"failed_hypotheses":["generated clients own authentication"],'
            '"memory_id":"'
            + episode_metadata.memory_id
            + '","query_sequence":["search_text:authentication"],'
            '"successful_paths":["src/http/routes.py","src/auth/service.py"],'
            '"summary":"A prior accepted turn followed route registration to '
            'the authentication service.","termination_status":"accepted",'
            '"usage_policy":"observed_strategy_history_only_never_code_evidence"}'
        ),
        estimated_tokens=1,
        priority=40,
    )

    repository_metadata = repository_memory.source.memory
    assert repository_metadata is not None
    decisions = (
        MemoryRecallDecision(
            retrieval_id=repository_metadata.retrieval_id,
            memory_id=repository_metadata.memory_id,
            scope=repository_metadata.scope,
            freshness=repository_metadata.freshness,
            staleness_reason=repository_metadata.staleness_reason,
            context_item_id=repository_memory.item_id,
        ),
        MemoryRecallDecision(
            retrieval_id=_digest("memory-ablation-stale-retrieval"),
            memory_id=_digest("memory-ablation-stale-memory"),
            scope=ContextMemoryScope.REPOSITORY,
            freshness=ContextMemoryFreshness.STALE,
            staleness_reason=ContextMemoryStalenessReason.SOURCE_MISSING,
            context_item_id=None,
        ),
        MemoryRecallDecision(
            retrieval_id=episode_metadata.retrieval_id,
            memory_id=episode_metadata.memory_id,
            scope=episode_metadata.scope,
            freshness=episode_metadata.freshness,
            context_item_id=episode_memory.item_id,
        ),
    )
    recall_report = create_memory_recall_report(
        query_digest=query_digest,
        decisions=decisions,
    )
    return create_memory_ablation_case(
        name="memory-recall-v1",
        budget_tokens=5_000,
        base_candidates=base_candidates,
        memory_candidates=(repository_memory, episode_memory),
        recall_report=recall_report,
        required_evidence_item_ids=context_case.required_evidence_item_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-case", type=Path, default=DEFAULT_CONTEXT_CASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    case = build_case(arguments.context_case)
    mode = "w" if arguments.force else "x"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(case.model_dump_json(indent=2))
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
