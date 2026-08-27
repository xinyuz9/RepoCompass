"""Deterministic candidate-to-graph and traceable-reading-path integration."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.structure.contracts import (
    PythonFileIndex,
    SourceLocation,
    StructuralVerificationStatus,
)
from repocompass.structure.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphNode,
    EvidenceNodeKind,
    create_evidence_graph,
    create_graph_edge,
)
from repocompass.structure.graph_builder import build_evidence_graph
from repocompass.structure.graph_slice import (
    MAX_GRAPH_SLICE_EDGES,
    MAX_GRAPH_SLICE_NODES,
    MAX_GRAPH_SLICE_SEEDS,
    ReadingPathError,
)
from repocompass.structure.python_index import (
    PythonIndexError,
    build_python_file_index,
)
from repocompass.structure.reading_path import (
    MAX_READING_TEXT_CHARACTERS,
    TraceableReadingPath,
    build_traceable_reading_path,
    create_traceable_claim,
)
from repocompass.tools.contracts import CommitSha, Sha256Digest
from repocompass.tools.gateway import ToolExecutionContext

INVESTIGATION_STRUCTURE_SCHEMA_VERSION = 1
MAX_CANDIDATE_STRUCTURE_FILES = 32
MAX_CANDIDATE_STRUCTURE_RECORDS = 900


class _StructuralArtifactModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class RelevantSymbol(_StructuralArtifactModel):
    """A candidate evidence item deterministically bound to one AST symbol."""

    schema_version: Literal[1] = INVESTIGATION_STRUCTURE_SCHEMA_VERSION
    binding_id: Sha256Digest
    entity_id: Sha256Digest
    node_id: Sha256Digest
    canonical_name: str = Field(min_length=1, max_length=2_048)
    path: str = Field(min_length=1, max_length=4_096)
    kind: EvidenceNodeKind
    verification_status: Literal[StructuralVerificationStatus.VERIFIED] = (
        StructuralVerificationStatus.VERIFIED
    )
    source_location: SourceLocation
    evidence_indices: tuple[int, ...] = Field(min_length=1, max_length=64)

    @field_validator("evidence_indices")
    @classmethod
    def require_canonical_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(index) is not int or index < 0 for index in value):
            raise ValueError("relevant-symbol evidence indices are invalid")
        if value != tuple(sorted(set(value))):
            raise ValueError("relevant-symbol evidence indices must be canonical")
        return value

    @model_validator(mode="after")
    def require_content_addressed_binding(self) -> "RelevantSymbol":
        expected = _relevant_symbol_id(
            entity_id=self.entity_id,
            node_id=self.node_id,
            canonical_name=self.canonical_name,
            path=self.path,
            kind=self.kind,
            source_location=self.source_location,
            evidence_indices=self.evidence_indices,
        )
        if self.binding_id != expected:
            raise ValueError("binding_id does not match relevant symbol contents")
        return self


class InvestigationStructure(_StructuralArtifactModel):
    """Stable graph and reading-path artifact for one evaluated candidate."""

    schema_version: Literal[1] = INVESTIGATION_STRUCTURE_SCHEMA_VERSION
    artifact_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    indexed_paths: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_CANDIDATE_STRUCTURE_FILES,
    )
    graph: EvidenceGraph
    reading_path: TraceableReadingPath
    relevant_symbols: tuple[RelevantSymbol, ...] = Field(
        default=(),
        max_length=64,
    )
    candidate_relation_assertion_ids: tuple[Sha256Digest, ...] = Field(
        default=(),
        max_length=64,
    )

    @model_validator(mode="after")
    def require_closed_content_addressed_artifact(self) -> "InvestigationStructure":
        if self.indexed_paths != tuple(sorted(set(self.indexed_paths))):
            raise ValueError("indexed paths must be unique and canonical")
        graph_paths = tuple(
            sorted(
                node.path
                for node in self.graph.nodes
                if node.kind is EvidenceNodeKind.FILE and node.path is not None
            )
        )
        if self.indexed_paths != graph_paths:
            raise ValueError("indexed paths do not match graph file nodes")
        if (
            self.graph.repository_id != self.repository_id
            or self.graph.commit_sha != self.commit_sha
            or self.reading_path.graph_slice.graph_id != self.graph.graph_id
            or self.reading_path.graph_slice.repository_id != self.repository_id
            or self.reading_path.graph_slice.commit_sha != self.commit_sha
        ):
            raise ValueError("structure artifact has mismatched graph provenance")
        if self.relevant_symbols != tuple(
            sorted(self.relevant_symbols, key=lambda item: item.binding_id)
        ):
            raise ValueError("relevant symbols must be canonical")
        if self.candidate_relation_assertion_ids != tuple(
            sorted(set(self.candidate_relation_assertion_ids))
        ):
            raise ValueError("candidate relation assertions must be canonical")

        nodes = {node.entity_id: node for node in self.graph.nodes}
        for binding in self.relevant_symbols:
            node = nodes.get(binding.entity_id)
            if (
                node is None
                or node.node_id != binding.node_id
                or node.canonical_name != binding.canonical_name
                or node.path != binding.path
                or node.kind is not binding.kind
                or not node.evidence
                or node.evidence[0].location != binding.source_location
            ):
                raise ValueError("relevant symbol does not match graph node")
        graph_assertions = {edge.assertion_id for edge in self.graph.edges}
        if not set(self.candidate_relation_assertion_ids).issubset(graph_assertions):
            raise ValueError("candidate relation assertion is outside graph")
        lineage_assertions = {
            assertion_id
            for lineage in self.reading_path.claim_lineages
            for assertion_id in lineage.supporting_assertion_ids
        }
        if lineage_assertions != set(self.candidate_relation_assertion_ids):
            raise ValueError("candidate relation assertions lack exact claim lineage")

        expected = _investigation_structure_id(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            indexed_paths=self.indexed_paths,
            graph=self.graph,
            reading_path=self.reading_path,
            relevant_symbols=self.relevant_symbols,
            candidate_relation_assertion_ids=self.candidate_relation_assertion_ids,
        )
        if self.artifact_id != expected:
            raise ValueError("artifact_id does not match structure contents")
        return self


def revalidate_investigation_structure(value: object) -> InvestigationStructure:
    """Restore canonical nested types and verify every content-addressed ID."""

    if not isinstance(value, InvestigationStructure):
        raise TypeError("value must be an InvestigationStructure")
    return InvestigationStructure.model_validate_json(
        value.model_dump_json(exclude_computed_fields=True, warnings=False)
    )


@dataclass(frozen=True, slots=True)
class StructureBuildIssue:
    """Sanitized deterministic defect converted to evaluator validation output."""

    code: str
    message: str
    path: str | None = None
    evidence_index: int | None = None
    relation_index: int | None = None


@dataclass(frozen=True, slots=True)
class StructureBuildResult:
    artifact: InvestigationStructure | None
    issues: tuple[StructureBuildIssue, ...]


def build_investigation_structure(
    *,
    repo_path: Path,
    objective: str,
    candidate: CandidateResult,
    valid_evidence_indices: frozenset[int],
) -> StructureBuildResult:
    """Build one bounded artifact without trusting model-proposed verification."""

    if not isinstance(repo_path, Path):
        raise TypeError("repo_path must be a Path")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be non-empty")
    if not isinstance(candidate, CandidateResult):
        raise TypeError("candidate must be CandidateResult")
    if not isinstance(valid_evidence_indices, frozenset) or any(
        type(index) is not int or index < 0 or index >= len(candidate.evidence)
        for index in valid_evidence_indices
    ):
        raise ValueError("valid_evidence_indices are invalid")

    repository = ToolExecutionContext(repo_path)
    issues: list[StructureBuildIssue] = []
    path_evidence: dict[str, list[int]] = {}
    for evidence_index in sorted(valid_evidence_indices):
        evidence = candidate.evidence[evidence_index]
        if evidence.path.endswith(".py"):
            path_evidence.setdefault(evidence.path, []).append(evidence_index)
    paths = tuple(sorted(path_evidence))
    if len(paths) > MAX_CANDIDATE_STRUCTURE_FILES:
        issues.append(
            StructureBuildIssue(
                code="structure_path_limit_exceeded",
                message=(
                    "Candidate Python evidence exceeds the bounded structural "
                    f"file limit of {MAX_CANDIDATE_STRUCTURE_FILES}."
                ),
            )
        )
        paths = paths[:MAX_CANDIDATE_STRUCTURE_FILES]

    indexes: list[PythonFileIndex] = []
    record_count = 0
    for path in paths:
        try:
            index = build_python_file_index(
                repository.repository_root,
                Path(path),
                repository_id=repository.repository_id,
                commit_sha=repository.commit_sha,
            )
        except PythonIndexError as error:
            issues.append(
                StructureBuildIssue(
                    code="structure_index_failed",
                    message=f"Python structure could not be indexed ({error.reason.value}).",
                    path=path,
                    evidence_index=path_evidence[path][0],
                )
            )
            continue
        next_count = (
            record_count
            + len(index.symbols)
            + len(index.imports)
            + len(index.references)
        )
        if next_count > MAX_CANDIDATE_STRUCTURE_RECORDS:
            issues.append(
                StructureBuildIssue(
                    code="structure_record_limit_exceeded",
                    message=(
                        "Candidate structure exceeds the bounded AST record limit "
                        f"of {MAX_CANDIDATE_STRUCTURE_RECORDS}."
                    ),
                    path=path,
                    evidence_index=path_evidence[path][0],
                )
            )
            continue
        indexes.append(index)
        record_count = next_count

    if not indexes:
        if candidate.relations:
            issues.append(
                StructureBuildIssue(
                    code="relation_structure_unavailable",
                    message="Candidate relations require validated Python structure.",
                    relation_index=0,
                )
            )
        return StructureBuildResult(artifact=None, issues=tuple(issues))

    try:
        graph = build_evidence_graph(tuple(indexes))
    except ValueError:
        issues.append(
            StructureBuildIssue(
                code="structure_graph_failed",
                message="Validated Python indexes could not form an unambiguous graph.",
            )
        )
        return StructureBuildResult(artifact=None, issues=tuple(issues))
    if (
        len(graph.nodes) > MAX_GRAPH_SLICE_NODES
        or len(graph.edges) > MAX_GRAPH_SLICE_EDGES
    ):
        issues.append(
            StructureBuildIssue(
                code="structure_graph_limit_exceeded",
                message="Candidate graph exceeds the bounded reading-path limits.",
            )
        )
        return StructureBuildResult(artifact=None, issues=tuple(issues))

    indexed_paths = tuple(
        index.path for index in sorted(indexes, key=lambda item: item.path)
    )
    indexed_path_set = set(indexed_paths)
    nodes_by_path: dict[str, list[EvidenceGraphNode]] = {}
    file_nodes: dict[str, EvidenceGraphNode] = {}
    for node in graph.nodes:
        if node.path is not None:
            nodes_by_path.setdefault(node.path, []).append(node)
        if node.kind is EvidenceNodeKind.FILE and node.path is not None:
            file_nodes[node.path] = node

    binding_indices: dict[str, set[int]] = {}
    seed_ids: set[str] = set()
    for evidence_index in sorted(valid_evidence_indices):
        evidence = candidate.evidence[evidence_index]
        if evidence.path not in indexed_path_set:
            continue
        file_node = file_nodes[evidence.path]
        if evidence.symbol is None:
            seed_ids.add(file_node.entity_id)
            continue
        matches = _resolve_symbol_on_path(nodes_by_path[evidence.path], evidence.symbol)
        if not matches:
            issues.append(
                StructureBuildIssue(
                    code="symbol_not_found",
                    message="Candidate symbol was not found in the indexed evidence path.",
                    path=evidence.path,
                    evidence_index=evidence_index,
                )
            )
            seed_ids.add(file_node.entity_id)
            continue
        if len(matches) > 1:
            issues.append(
                StructureBuildIssue(
                    code="symbol_ambiguous",
                    message="Candidate symbol is ambiguous in the indexed evidence path.",
                    path=evidence.path,
                    evidence_index=evidence_index,
                )
            )
            seed_ids.add(file_node.entity_id)
            continue
        node = matches[0]
        location = node.evidence[0].location
        line_end = evidence.line_end or evidence.line_start
        if (
            evidence.line_start is None
            or line_end is None
            or line_end < location.line_start
            or evidence.line_start > location.line_end
        ):
            issues.append(
                StructureBuildIssue(
                    code="symbol_line_mismatch",
                    message="Candidate lines do not overlap the resolved symbol definition.",
                    path=evidence.path,
                    evidence_index=evidence_index,
                )
            )
            seed_ids.add(file_node.entity_id)
            continue
        binding_indices.setdefault(node.entity_id, set()).add(evidence_index)
        seed_ids.add(node.entity_id)

    relevant_symbols = tuple(
        sorted(
            (
                _create_relevant_symbol(
                    node=_node_by_entity_id(graph, entity_id),
                    evidence_indices=tuple(sorted(indices)),
                )
                for entity_id, indices in binding_indices.items()
            ),
            key=lambda item: item.binding_id,
        )
    )

    relation_assertions: set[str] = set()
    relation_claims = []
    for relation_index, relation in enumerate(candidate.relations):
        if any(
            index not in valid_evidence_indices
            for index in relation.supporting_evidence_indices
        ):
            issues.append(
                StructureBuildIssue(
                    code="relation_evidence_invalid",
                    message="Candidate relation references invalid evidence.",
                    evidence_index=relation.supporting_evidence_indices[0],
                    relation_index=relation_index,
                )
            )
            continue
        source_matches = _resolve_graph_selector(graph, relation.source)
        target_matches = _resolve_graph_selector(graph, relation.target)
        if not source_matches or not target_matches:
            issues.append(
                StructureBuildIssue(
                    code="relation_endpoint_not_found",
                    message="Candidate relation endpoint was not found in the graph.",
                    evidence_index=relation.supporting_evidence_indices[0],
                    relation_index=relation_index,
                )
            )
            continue
        if len(source_matches) > 1 or len(target_matches) > 1:
            issues.append(
                StructureBuildIssue(
                    code="relation_endpoint_ambiguous",
                    message="Candidate relation endpoint is ambiguous in the graph.",
                    evidence_index=relation.supporting_evidence_indices[0],
                    relation_index=relation_index,
                )
            )
            continue
        source, target = source_matches[0], target_matches[0]
        existing = next(
            (
                edge
                for edge in graph.edges
                if edge.relation is relation.relation
                and edge.source_entity_id == source.entity_id
                and edge.target_entity_id == target.entity_id
            ),
            None,
        )
        if existing is None:
            proposed = create_graph_edge(
                repository_id=graph.repository_id,
                commit_sha=graph.commit_sha,
                relation=relation.relation,
                source_entity_id=source.entity_id,
                target_entity_id=target.entity_id,
                verification_status=StructuralVerificationStatus.INFERRED,
                explanation=relation.explanation,
            )
            try:
                graph = create_evidence_graph(
                    repository_id=graph.repository_id,
                    commit_sha=graph.commit_sha,
                    nodes=graph.nodes,
                    edges=(*graph.edges, proposed),
                )
            except ValueError:
                issues.append(
                    StructureBuildIssue(
                        code="relation_incompatible",
                        message="Candidate relation is incompatible with graph endpoints.",
                        evidence_index=relation.supporting_evidence_indices[0],
                        relation_index=relation_index,
                    )
                )
                continue
            edge = proposed
        else:
            edge = existing
        relation_assertions.add(edge.assertion_id)
        seed_ids.update((source.entity_id, target.entity_id))
        relation_claims.append(
            create_traceable_claim(
                text=(
                    f"Structural relation: {source.canonical_name} "
                    f"{relation.relation.value} {target.canonical_name}."
                ),
                supporting_assertion_ids=(edge.assertion_id,),
                explanation=(
                    relation.explanation
                    if edge.verification_status is StructuralVerificationStatus.INFERRED
                    else None
                ),
            )
        )

    if len(seed_ids) > MAX_GRAPH_SLICE_SEEDS:
        issues.append(
            StructureBuildIssue(
                code="structure_seed_limit_exceeded",
                message=(
                    "Relevant entities exceed the bounded reading-path seed limit "
                    f"of {MAX_GRAPH_SLICE_SEEDS}."
                ),
            )
        )
        return StructureBuildResult(artifact=None, issues=tuple(issues))
    try:
        reading_path = build_traceable_reading_path(
            graph,
            objective=_safe_reading_objective(objective),
            seed_entity_ids=tuple(sorted(seed_ids)),
            claims=tuple(relation_claims),
            include_inferred=any(
                edge.verification_status is StructuralVerificationStatus.INFERRED
                and edge.assertion_id in relation_assertions
                for edge in graph.edges
            ),
            max_depth=8,
            max_nodes=MAX_GRAPH_SLICE_NODES,
            max_edges=MAX_GRAPH_SLICE_EDGES,
        )
    except (ReadingPathError, ValueError):
        issues.append(
            StructureBuildIssue(
                code="reading_path_failed",
                message="Candidate relations could not form a complete bounded lineage.",
            )
        )
        return StructureBuildResult(artifact=None, issues=tuple(issues))

    ordered_assertions = tuple(sorted(relation_assertions))
    artifact_id = _investigation_structure_id(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        indexed_paths=indexed_paths,
        graph=graph,
        reading_path=reading_path,
        relevant_symbols=relevant_symbols,
        candidate_relation_assertion_ids=ordered_assertions,
    )
    artifact = InvestigationStructure(
        artifact_id=artifact_id,
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        indexed_paths=indexed_paths,
        graph=graph,
        reading_path=reading_path,
        relevant_symbols=relevant_symbols,
        candidate_relation_assertion_ids=ordered_assertions,
    )
    return StructureBuildResult(artifact=artifact, issues=tuple(issues))


def _resolve_symbol_on_path(
    nodes: list[EvidenceGraphNode], selector: str
) -> tuple[EvidenceGraphNode, ...]:
    exact = tuple(node for node in nodes if node.canonical_name == selector)
    if exact:
        return exact
    return tuple(
        node for node in nodes if node.canonical_name.rsplit(".", 1)[-1] == selector
    )


def _resolve_graph_selector(
    graph: EvidenceGraph, selector: str
) -> tuple[EvidenceGraphNode, ...]:
    by_id = tuple(node for node in graph.nodes if node.entity_id == selector)
    if by_id:
        return by_id
    exact = tuple(node for node in graph.nodes if node.canonical_name == selector)
    if exact:
        return exact
    if "." in selector:
        return ()
    return tuple(
        node
        for node in graph.nodes
        if node.canonical_name.rsplit(".", 1)[-1] == selector
    )


def _node_by_entity_id(graph: EvidenceGraph, entity_id: str) -> EvidenceGraphNode:
    return graph.node_by_entity_id(entity_id)


def _create_relevant_symbol(
    *, node: EvidenceGraphNode, evidence_indices: tuple[int, ...]
) -> RelevantSymbol:
    if node.path is None or not node.evidence:
        raise ValueError("relevant symbol requires source-backed graph node")
    source_location = node.evidence[0].location
    binding_id = _relevant_symbol_id(
        entity_id=node.entity_id,
        node_id=node.node_id,
        canonical_name=node.canonical_name,
        path=node.path,
        kind=node.kind,
        source_location=source_location,
        evidence_indices=evidence_indices,
    )
    return RelevantSymbol(
        binding_id=binding_id,
        entity_id=node.entity_id,
        node_id=node.node_id,
        canonical_name=node.canonical_name,
        path=node.path,
        kind=node.kind,
        source_location=source_location,
        evidence_indices=evidence_indices,
    )


def _safe_reading_objective(objective: str) -> str:
    normalized = " ".join(objective.split()).replace("\x00", "")
    bounded = normalized[:MAX_READING_TEXT_CHARACTERS].rstrip()
    return bounded or "Repository investigation"


def _relevant_symbol_id(
    *,
    entity_id: str,
    node_id: str,
    canonical_name: str,
    path: str,
    kind: EvidenceNodeKind,
    source_location: SourceLocation,
    evidence_indices: tuple[int, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:relevant-symbol:v1",
        {
            "entity_id": entity_id,
            "node_id": node_id,
            "canonical_name": canonical_name,
            "path": path,
            "kind": kind.value,
            "source_location": source_location.model_dump(mode="json"),
            "evidence_indices": evidence_indices,
        },
    )


def _investigation_structure_id(
    *,
    repository_id: str,
    commit_sha: str | None,
    indexed_paths: tuple[str, ...],
    graph: EvidenceGraph,
    reading_path: TraceableReadingPath,
    relevant_symbols: tuple[RelevantSymbol, ...],
    candidate_relation_assertion_ids: tuple[str, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:investigation-structure:v1",
        {
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "indexed_paths": indexed_paths,
            "graph": graph.model_dump(mode="json"),
            "reading_path": reading_path.model_dump(mode="json"),
            "relevant_symbols": [
                item.model_dump(mode="json") for item in relevant_symbols
            ],
            "candidate_relation_assertion_ids": candidate_relation_assertion_ids,
        },
    )


def _canonical_hash(namespace: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()
