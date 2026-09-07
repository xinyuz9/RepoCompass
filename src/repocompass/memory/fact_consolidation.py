"""Promote only re-derived static Python facts, never model prose, into memory."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.context import RepositoryLocation
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.lifecycle import ExecutionStatus
from repocompass.agent.result import revalidate_final_result
from repocompass.agent.runner import RunOutcome
from repocompass.agent.structural_artifacts import MAX_CANDIDATE_STRUCTURE_RECORDS
from repocompass.memory.contracts import (
    MemoryFactKind,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    create_repository_memory,
    create_repository_memory_source,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.revalidation import repository_snapshot_is_clean
from repocompass.memory.store import (
    MAX_GENERATED_FACTS_PER_WRITE,
    MemoryStore,
    MemoryStoreIntegrityError,
)
from repocompass.structure.contracts import StructuralVerificationStatus
from repocompass.structure.evidence_graph import EvidenceNodeKind
from repocompass.structure.graph_builder import build_evidence_graph
from repocompass.structure.python_index import PythonIndexError, build_python_file_index
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, Sha256Digest
from repocompass.tools.gateway import ToolExecutionContext


class RepositoryConsolidationResult(CheckpointSafeModel):
    """Body-free, bounded trace projection; IDs are live facts or tombstones."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    policy: Literal["accepted_static_python_v1"] = "accepted_static_python_v1"
    reason: Literal[
        "validated_structure",
        "not_accepted",
        "no_structure",
        "snapshot_changed",
        "source_unavailable",
        "structure_mismatch",
        "structure_limit",
        "no_eligible_facts",
    ]
    created_ids: tuple[Sha256Digest, ...] = Field(default=(), max_length=16)
    existing_ids: tuple[Sha256Digest, ...] = Field(default=(), max_length=16)
    tombstoned_ids: tuple[Sha256Digest, ...] = Field(default=(), max_length=16)
    truncated: bool = False

    @model_validator(mode="after")
    def require_consistent_ids(self):
        ids = self.created_ids + self.existing_ids + self.tombstoned_ids
        if len(ids) > MAX_GENERATED_FACTS_PER_WRITE or len(set(ids)) != len(ids):
            raise ValueError("fact consolidation identities exceed the bound or repeat")
        if self.reason != "validated_structure" and (ids or self.truncated):
            raise ValueError("skipped consolidation cannot claim writes")
        return self


def consolidate_repository_facts(
    store: MemoryStore,
    *,
    context: ToolExecutionContext,
    outcome: RunOutcome,
    created_at_ms: int,
) -> RepositoryConsolidationResult:
    """Re-index the accepted candidate's bounded files without executing code."""
    if not isinstance(store, MemoryStore) or not isinstance(
        context, ToolExecutionContext
    ):
        raise TypeError("fact consolidation requires a trusted store and context")
    if not isinstance(outcome, RunOutcome):
        raise TypeError("outcome must be a RunOutcome")
    if (
        type(created_at_ms) is not int
        or not 0 <= created_at_ms <= MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError("created_at_ms must be a safe timestamp")
    if (
        outcome.execution_status is not ExecutionStatus.COMPLETED
        or outcome.termination_status != "accepted"
    ):
        return RepositoryConsolidationResult(reason="not_accepted")
    result = revalidate_final_result(outcome.final_result)
    structure = result.structure
    if structure is None:
        return RepositoryConsolidationResult(reason="no_structure")
    if (
        context.commit_sha is None
        or context.commit_sha != structure.commit_sha
        or context.repository_id != structure.repository_id
        or not repository_snapshot_is_clean(context)
    ):
        return RepositoryConsolidationResult(reason="snapshot_changed")
    indexes = []
    record_count = 0
    try:
        for path in structure.indexed_paths:
            index = build_python_file_index(
                context.repository_root,
                Path(path),
                repository_id=context.repository_id,
                commit_sha=context.commit_sha,
            )
            record_count += (
                len(index.symbols) + len(index.imports) + len(index.references)
            )
            if record_count > MAX_CANDIDATE_STRUCTURE_RECORDS:
                return RepositoryConsolidationResult(reason="structure_limit")
            indexes.append(index)
        graph = build_evidence_graph(tuple(indexes))
    except (PythonIndexError, ValueError, OSError):
        return RepositoryConsolidationResult(reason="source_unavailable")

    nodes = {node.entity_id: node for node in graph.nodes}
    original_nodes = {node.entity_id: node for node in structure.graph.nodes}
    # Every supplied file anchor must still match fresh AST-derived bytes.
    for node in structure.graph.nodes:
        if node.kind is EvidenceNodeKind.FILE and nodes.get(node.entity_id) != node:
            return RepositoryConsolidationResult(reason="structure_mismatch")
    eligible = []
    selected_ids = {binding.entity_id for binding in structure.relevant_symbols}
    selected_ids.update(
        node.entity_id
        for node in structure.graph.nodes
        if node.kind is EvidenceNodeKind.FILE
    )
    for entity_id in sorted(selected_ids):
        node = nodes.get(entity_id)
        if node is None or node != original_nodes[entity_id] or not node.evidence:
            return RepositoryConsolidationResult(reason="structure_mismatch")
        content = (
            f"Static Python {node.kind.value} definition: {node.canonical_name} "
            f"in {node.path}, lines {node.evidence[0].location.line_start}-"
            f"{node.evidence[0].location.line_end}. This states syntax, not runtime behavior."
        )
        eligible.append(
            (
                MemoryFactKind.ARCHITECTURE_FACT,
                content,
                node.evidence,
                (),
                node.canonical_name,
            )
        )

    edges = {edge.assertion_id: edge for edge in graph.edges}
    selected_assertions = set(structure.candidate_relation_assertion_ids)
    for edge in structure.graph.edges:
        if (
            edge.assertion_id not in selected_assertions
            or edge.verification_status is not StructuralVerificationStatus.VERIFIED
        ):
            continue
        if edges.get(edge.assertion_id) != edge:
            return RepositoryConsolidationResult(reason="structure_mismatch")
        source, target = nodes[edge.source_entity_id], nodes[edge.target_entity_id]
        # Explicitly retain endpoint sources as well as the relation occurrence.
        evidence = tuple(
            {
                item.evidence_id: item
                for item in (*edge.evidence, *source.evidence, *target.evidence)
            }.values()
        )
        if not evidence or len(evidence) > 32:
            continue
        content = (
            f"Static Python relation: {source.canonical_name} {edge.relation.value} "
            f"{target.canonical_name}. Verified by the bounded AST resolver; "
            "this is not proof of runtime execution or arbitrary behavioral claims."
        )
        eligible.append(
            (
                MemoryFactKind.SYMBOL_RELATION,
                content,
                evidence,
                (edge.assertion_id,),
                source.canonical_name,
            )
        )
    if not eligible:
        return RepositoryConsolidationResult(reason="no_eligible_facts")
    # Prefer requested relations, then definitions; deterministic and independent of LLM wording.
    eligible.sort(
        key=lambda item: (item[0] is not MemoryFactKind.SYMBOL_RELATION, item[1])
    )
    identity = resolve_memory_repository_identity(context)
    memories = []
    for kind, content, evidence, assertions, label in eligible[
        :MAX_GENERATED_FACTS_PER_WRITE
    ]:
        claim_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        sources = {
            source.source_id: source
            for source in (
                create_repository_memory_source(
                    location=RepositoryLocation(
                        repository_id=context.repository_id,
                        commit_sha=context.commit_sha,
                        path=item.location.path,
                        line_start=item.location.line_start,
                        line_end=item.location.line_end,
                    ),
                    source_digest=item.location.source_digest,
                    evidence_claim_digest=claim_digest,
                    supporting_assertion_ids=assertions,
                )
                for item in evidence
            )
        }
        memories.append(
            create_repository_memory(
                identity=identity,
                commit_sha=context.commit_sha,
                kind=kind,
                title=("Static Python " + " ".join(label.split()))[:512].rstrip(),
                content=content,
                verification_status=MemoryVerificationStatus.VERIFIED,
                write_authority=MemoryWriteAuthority.VALIDATED_EVIDENCE,
                sources=tuple(sources[key] for key in sorted(sources)),
                tags=("automatic-static-python-v1", "python", "structure"),
                created_at_ms=created_at_ms,
                verified_at_ms=created_at_ms,
            )
        )
    current = ToolExecutionContext(context.repository_root)
    if (
        current.commit_sha != context.commit_sha
        or current.repository_id != context.repository_id
    ):
        return RepositoryConsolidationResult(reason="snapshot_changed")
    try:
        written = store.put_generated_repository_memories(
            tuple(memories), context=current, run_id=outcome.trace.run_id
        )
    except MemoryStoreIntegrityError:
        # A source can change after indexing. The batch transaction rolls back.
        return RepositoryConsolidationResult(reason="snapshot_changed")
    return RepositoryConsolidationResult(
        reason="validated_structure",
        created_ids=written.created_ids,
        existing_ids=written.existing_ids,
        tombstoned_ids=written.tombstoned_ids,
        truncated=len(eligible) > MAX_GENERATED_FACTS_PER_WRITE,
    )
