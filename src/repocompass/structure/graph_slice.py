"""Bounded deterministic slices over immutable Evidence Graphs."""

import hashlib
import json
from collections import deque
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.structure.contracts import StructuralVerificationStatus
from repocompass.structure.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    canonical_graph_edges,
    canonical_graph_nodes,
)
from repocompass.tools.contracts import CommitSha, Sha256Digest

GRAPH_SLICE_SCHEMA_VERSION = 1
MAX_GRAPH_SLICE_SEEDS = 32
MAX_GRAPH_SLICE_DEPTH = 8
MAX_GRAPH_SLICE_NODES = 2_000
MAX_GRAPH_SLICE_EDGES = 5_000


class GraphTraversalDirection(StrEnum):
    """Direction used when expanding relations from each selected entity."""

    OUTBOUND = "outbound"
    INBOUND = "inbound"
    BOTH = "both"


class GraphSliceTruncationReason(StrEnum):
    """Stable reasons why a bounded slice omitted an eligible relation."""

    DEPTH_LIMIT = "depth_limit"
    NODE_LIMIT = "node_limit"
    EDGE_LIMIT = "edge_limit"


class ReadingPathErrorReason(StrEnum):
    """Non-sensitive failures shared by slicing and reading-path assembly."""

    EMPTY_SEEDS = "empty_seeds"
    UNKNOWN_SEED = "unknown_seed"
    AMBIGUOUS_SEED = "ambiguous_seed"
    DUPLICATE_SEED = "duplicate_seed"
    SEED_LIMIT = "seed_limit"
    INVALID_LIMIT = "invalid_limit"
    NO_SOURCE_STEPS = "no_source_steps"
    UNKNOWN_ASSERTION = "unknown_assertion"
    REJECTED_ASSERTION = "rejected_assertion"
    ASSERTION_OUTSIDE_SLICE = "assertion_outside_slice"
    INFERRED_CLAIM_EXPLANATION_REQUIRED = "inferred_claim_explanation_required"
    VERIFIED_CLAIM_EXPLANATION = "verified_claim_explanation"


class ReadingPathError(ValueError):
    """Fail closed with a stable reason and without repository source text."""

    def __init__(self, reason: ReadingPathErrorReason) -> None:
        if not isinstance(reason, ReadingPathErrorReason):
            raise TypeError("reason must be ReadingPathErrorReason")
        self.reason = reason
        super().__init__(f"reading path failed: {reason.value}")


class _GraphSliceModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class GraphNodeDepth(_GraphSliceModel):
    """Shortest bounded traversal distance for one selected graph node."""

    schema_version: Literal[1] = GRAPH_SLICE_SCHEMA_VERSION
    entity_id: Sha256Digest
    depth: int = Field(ge=0, le=MAX_GRAPH_SLICE_DEPTH)
    is_seed: bool

    @model_validator(mode="after")
    def require_seed_depth(self) -> "GraphNodeDepth":
        if self.is_seed != (self.depth == 0):
            raise ValueError("only depth-zero nodes may be marked as seeds")
        return self


class EvidenceGraphSlice(_GraphSliceModel):
    """Canonical accepted subgraph selected under explicit traversal bounds."""

    schema_version: Literal[1] = GRAPH_SLICE_SCHEMA_VERSION
    slice_id: Sha256Digest
    graph_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    seed_entity_ids: tuple[Sha256Digest, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_SLICE_SEEDS,
    )
    traversal: GraphTraversalDirection
    include_inferred: bool
    max_depth: int = Field(ge=0, le=MAX_GRAPH_SLICE_DEPTH)
    max_nodes: int = Field(ge=1, le=MAX_GRAPH_SLICE_NODES)
    max_edges: int = Field(ge=0, le=MAX_GRAPH_SLICE_EDGES)
    truncated: bool
    truncation_reasons: tuple[GraphSliceTruncationReason, ...] = Field(
        default=(),
        max_length=len(GraphSliceTruncationReason),
    )
    nodes: tuple[EvidenceGraphNode, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_SLICE_NODES,
    )
    edges: tuple[EvidenceGraphEdge, ...] = Field(
        default=(),
        max_length=MAX_GRAPH_SLICE_EDGES,
    )
    node_depths: tuple[GraphNodeDepth, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_SLICE_NODES,
    )

    @model_validator(mode="after")
    def require_canonical_closed_content_addressed_slice(self) -> "EvidenceGraphSlice":
        if self.seed_entity_ids != tuple(sorted(self.seed_entity_ids)) or len(
            self.seed_entity_ids
        ) != len(set(self.seed_entity_ids)):
            raise ValueError("seed entity IDs must be unique and canonical")
        if self.nodes != canonical_graph_nodes(self.nodes):
            raise ValueError("slice nodes must be unique and canonical")
        if self.edges != canonical_graph_edges(self.edges):
            raise ValueError("slice edges must be unique and canonical")
        canonical_depths = tuple(
            sorted(self.node_depths, key=lambda item: item.entity_id)
        )
        if self.node_depths != canonical_depths:
            raise ValueError("node depths must be unique and canonical")
        if self.truncation_reasons != _canonical_truncation_reasons(
            self.truncation_reasons
        ):
            raise ValueError("truncation reasons must be unique and canonical")
        if self.truncated != bool(self.truncation_reasons):
            raise ValueError("truncated must match truncation reasons")
        if len(self.nodes) > self.max_nodes or len(self.edges) > self.max_edges:
            raise ValueError("slice contents exceed declared limits")
        if len(self.seed_entity_ids) > self.max_nodes:
            raise ValueError("seed count exceeds node limit")

        nodes_by_id = {node.entity_id: node for node in self.nodes}
        if len(nodes_by_id) != len(self.nodes):
            raise ValueError("slice entity IDs must be unique")
        depths_by_id = {item.entity_id: item for item in self.node_depths}
        if len(depths_by_id) != len(self.node_depths):
            raise ValueError("slice node depths must be unique")
        if set(nodes_by_id) != set(depths_by_id):
            raise ValueError("slice node depths must cover exactly the selected nodes")
        if not set(self.seed_entity_ids).issubset(nodes_by_id):
            raise ValueError("slice seed is not present in selected nodes")
        for entity_id, depth in depths_by_id.items():
            if depth.depth > self.max_depth:
                raise ValueError("node depth exceeds declared depth limit")
            if depth.is_seed != (entity_id in self.seed_entity_ids):
                raise ValueError("node seed marker does not match seed identity")
        for node in self.nodes:
            if (
                node.repository_id != self.repository_id
                or node.commit_sha != self.commit_sha
            ):
                raise ValueError("slice node has mismatched snapshot")
        assertion_ids: set[str] = set()
        for edge in self.edges:
            if edge.assertion_id in assertion_ids:
                raise ValueError("slice assertion IDs must be unique")
            assertion_ids.add(edge.assertion_id)
            if (
                edge.repository_id != self.repository_id
                or edge.commit_sha != self.commit_sha
            ):
                raise ValueError("slice edge has mismatched snapshot")
            if (
                edge.source_entity_id not in nodes_by_id
                or edge.target_entity_id not in nodes_by_id
            ):
                raise ValueError("slice edge has a dangling endpoint")
            if edge.verification_status is StructuralVerificationStatus.REJECTED:
                raise ValueError("rejected edge cannot enter an evidence graph slice")
            if (
                not self.include_inferred
                and edge.verification_status is StructuralVerificationStatus.INFERRED
            ):
                raise ValueError("inferred edge is disabled for this slice")

        expected = _slice_id(
            graph_id=self.graph_id,
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            seed_entity_ids=self.seed_entity_ids,
            traversal=self.traversal,
            include_inferred=self.include_inferred,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            truncated=self.truncated,
            truncation_reasons=self.truncation_reasons,
            nodes=self.nodes,
            edges=self.edges,
            node_depths=self.node_depths,
        )
        if self.slice_id != expected:
            raise ValueError("slice_id does not match graph slice contents")
        return self

    def depth_for(self, entity_id: str) -> int:
        """Return the persisted shortest distance for one selected entity."""

        for item in self.node_depths:
            if item.entity_id == entity_id:
                return item.depth
        raise KeyError(entity_id)


def resolve_seed_entity_ids(
    graph: EvidenceGraph,
    selectors: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve exact entity IDs or canonical names and reject ambiguity."""

    if not isinstance(graph, EvidenceGraph):
        raise TypeError("graph must be EvidenceGraph")
    if not isinstance(selectors, tuple):
        raise TypeError("selectors must be a tuple")
    if not selectors:
        raise ReadingPathError(ReadingPathErrorReason.EMPTY_SEEDS)
    if len(selectors) > MAX_GRAPH_SLICE_SEEDS:
        raise ReadingPathError(ReadingPathErrorReason.SEED_LIMIT)
    nodes_by_id = {node.entity_id: node for node in graph.nodes}
    resolved: list[str] = []
    for selector in selectors:
        if (
            not isinstance(selector, str)
            or not selector
            or selector.strip() != selector
        ):
            raise ReadingPathError(ReadingPathErrorReason.UNKNOWN_SEED)
        if selector in nodes_by_id:
            matches = (nodes_by_id[selector],)
        else:
            matches = tuple(
                node for node in graph.nodes if node.canonical_name == selector
            )
        if not matches:
            raise ReadingPathError(ReadingPathErrorReason.UNKNOWN_SEED)
        if len(matches) > 1:
            raise ReadingPathError(ReadingPathErrorReason.AMBIGUOUS_SEED)
        resolved.append(matches[0].entity_id)
    if len(resolved) != len(set(resolved)):
        raise ReadingPathError(ReadingPathErrorReason.DUPLICATE_SEED)
    return tuple(sorted(resolved))


def slice_evidence_graph(
    graph: EvidenceGraph,
    *,
    seed_entity_ids: tuple[str, ...],
    traversal: GraphTraversalDirection = GraphTraversalDirection.BOTH,
    include_inferred: bool = False,
    max_depth: int = 3,
    max_nodes: int = 100,
    max_edges: int = 200,
) -> EvidenceGraphSlice:
    """Select a deterministic accepted BFS slice under three hard limits."""

    if not isinstance(graph, EvidenceGraph):
        raise TypeError("graph must be EvidenceGraph")
    if not isinstance(seed_entity_ids, tuple):
        raise TypeError("seed_entity_ids must be a tuple")
    if not isinstance(traversal, GraphTraversalDirection):
        raise TypeError("traversal must be GraphTraversalDirection")
    if type(include_inferred) is not bool:
        raise TypeError("include_inferred must be bool")
    _validate_limits(max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges)
    if not seed_entity_ids:
        raise ReadingPathError(ReadingPathErrorReason.EMPTY_SEEDS)
    if len(seed_entity_ids) > MAX_GRAPH_SLICE_SEEDS:
        raise ReadingPathError(ReadingPathErrorReason.SEED_LIMIT)
    if len(seed_entity_ids) != len(set(seed_entity_ids)):
        raise ReadingPathError(ReadingPathErrorReason.DUPLICATE_SEED)
    if len(seed_entity_ids) > max_nodes:
        raise ReadingPathError(ReadingPathErrorReason.SEED_LIMIT)

    nodes_by_id = {node.entity_id: node for node in graph.nodes}
    if any(entity_id not in nodes_by_id for entity_id in seed_entity_ids):
        raise ReadingPathError(ReadingPathErrorReason.UNKNOWN_SEED)
    ordered_seeds = tuple(sorted(seed_entity_ids))
    eligible_edges = tuple(
        edge
        for edge in graph.edges
        if edge.verification_status is not StructuralVerificationStatus.REJECTED
        and (
            include_inferred
            or edge.verification_status is StructuralVerificationStatus.VERIFIED
        )
    )
    adjacency = _build_adjacency(eligible_edges, traversal)

    selected_nodes = {entity_id: nodes_by_id[entity_id] for entity_id in ordered_seeds}
    selected_edges: dict[str, EvidenceGraphEdge] = {}
    depths = {entity_id: 0 for entity_id in ordered_seeds}
    queue: deque[str] = deque(ordered_seeds)
    truncation_reasons: set[GraphSliceTruncationReason] = set()
    while queue:
        current_id = queue.popleft()
        current_depth = depths[current_id]
        for edge, neighbor_id in adjacency.get(current_id, ()):
            if edge.assertion_id in selected_edges:
                continue
            neighbor_selected = neighbor_id in selected_nodes
            if not neighbor_selected and current_depth >= max_depth:
                truncation_reasons.add(GraphSliceTruncationReason.DEPTH_LIMIT)
                continue
            if len(selected_edges) >= max_edges:
                truncation_reasons.add(GraphSliceTruncationReason.EDGE_LIMIT)
                continue
            if not neighbor_selected and len(selected_nodes) >= max_nodes:
                truncation_reasons.add(GraphSliceTruncationReason.NODE_LIMIT)
                continue
            selected_edges[edge.assertion_id] = edge
            if not neighbor_selected:
                selected_nodes[neighbor_id] = nodes_by_id[neighbor_id]
                depths[neighbor_id] = current_depth + 1
                queue.append(neighbor_id)

    ordered_nodes = canonical_graph_nodes(tuple(selected_nodes.values()))
    ordered_edges = canonical_graph_edges(tuple(selected_edges.values()))
    ordered_depths = tuple(
        GraphNodeDepth(
            entity_id=entity_id,
            depth=depths[entity_id],
            is_seed=entity_id in ordered_seeds,
        )
        for entity_id in sorted(selected_nodes)
    )
    ordered_reasons = _canonical_truncation_reasons(tuple(truncation_reasons))
    return _create_graph_slice(
        graph_id=graph.graph_id,
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        seed_entity_ids=ordered_seeds,
        traversal=traversal,
        include_inferred=include_inferred,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_edges=max_edges,
        truncated=bool(ordered_reasons),
        truncation_reasons=ordered_reasons,
        nodes=ordered_nodes,
        edges=ordered_edges,
        node_depths=ordered_depths,
    )


def _create_graph_slice(
    *,
    graph_id: str,
    repository_id: str,
    commit_sha: str | None,
    seed_entity_ids: tuple[str, ...],
    traversal: GraphTraversalDirection,
    include_inferred: bool,
    max_depth: int,
    max_nodes: int,
    max_edges: int,
    truncated: bool,
    truncation_reasons: tuple[GraphSliceTruncationReason, ...],
    nodes: tuple[EvidenceGraphNode, ...],
    edges: tuple[EvidenceGraphEdge, ...],
    node_depths: tuple[GraphNodeDepth, ...],
) -> EvidenceGraphSlice:
    payload = {
        "graph_id": graph_id,
        "repository_id": repository_id,
        "commit_sha": commit_sha,
        "seed_entity_ids": seed_entity_ids,
        "traversal": traversal,
        "include_inferred": include_inferred,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "truncated": truncated,
        "truncation_reasons": truncation_reasons,
        "nodes": nodes,
        "edges": edges,
        "node_depths": node_depths,
    }
    return EvidenceGraphSlice(
        slice_id=_slice_id(**payload),
        **payload,
    )


def _build_adjacency(
    edges: tuple[EvidenceGraphEdge, ...],
    traversal: GraphTraversalDirection,
) -> dict[str, tuple[tuple[EvidenceGraphEdge, str], ...]]:
    mutable: dict[str, list[tuple[EvidenceGraphEdge, str]]] = {}
    for edge in edges:
        if traversal in {
            GraphTraversalDirection.OUTBOUND,
            GraphTraversalDirection.BOTH,
        }:
            mutable.setdefault(edge.source_entity_id, []).append(
                (edge, edge.target_entity_id)
            )
        if traversal in {GraphTraversalDirection.INBOUND, GraphTraversalDirection.BOTH}:
            mutable.setdefault(edge.target_entity_id, []).append(
                (edge, edge.source_entity_id)
            )
    return {
        entity_id: tuple(
            sorted(
                relations,
                key=lambda item: (
                    0
                    if item[0].verification_status
                    is StructuralVerificationStatus.VERIFIED
                    else 1,
                    item[0].relation.value,
                    item[1],
                    item[0].assertion_id,
                ),
            )
        )
        for entity_id, relations in mutable.items()
    }


def traversal_predecessor_candidates(
    graph_slice: EvidenceGraphSlice,
    entity_id: str,
) -> tuple[tuple[EvidenceGraphEdge, str], ...]:
    """Return deterministic depth-decreasing edges for reading-step lineage."""

    depth = graph_slice.depth_for(entity_id)
    if depth == 0:
        return ()
    candidates: list[tuple[EvidenceGraphEdge, str]] = []
    for edge in graph_slice.edges:
        predecessor_id: str | None = None
        if (
            graph_slice.traversal
            in {GraphTraversalDirection.OUTBOUND, GraphTraversalDirection.BOTH}
            and edge.target_entity_id == entity_id
        ):
            predecessor_id = edge.source_entity_id
        elif (
            graph_slice.traversal
            in {GraphTraversalDirection.INBOUND, GraphTraversalDirection.BOTH}
            and edge.source_entity_id == entity_id
        ):
            predecessor_id = edge.target_entity_id
        if (
            predecessor_id is not None
            and graph_slice.depth_for(predecessor_id) == depth - 1
        ):
            candidates.append((edge, predecessor_id))
    return tuple(candidates)


def _validate_limits(*, max_depth: int, max_nodes: int, max_edges: int) -> None:
    if (
        type(max_depth) is not int
        or type(max_nodes) is not int
        or type(max_edges) is not int
        or not 0 <= max_depth <= MAX_GRAPH_SLICE_DEPTH
        or not 1 <= max_nodes <= MAX_GRAPH_SLICE_NODES
        or not 0 <= max_edges <= MAX_GRAPH_SLICE_EDGES
    ):
        raise ReadingPathError(ReadingPathErrorReason.INVALID_LIMIT)


def _canonical_truncation_reasons(
    reasons: tuple[GraphSliceTruncationReason, ...],
) -> tuple[GraphSliceTruncationReason, ...]:
    reason_set = set(reasons)
    return tuple(
        reason for reason in GraphSliceTruncationReason if reason in reason_set
    )


def _slice_id(
    *,
    graph_id: str,
    repository_id: str,
    commit_sha: str | None,
    seed_entity_ids: tuple[str, ...],
    traversal: GraphTraversalDirection,
    include_inferred: bool,
    max_depth: int,
    max_nodes: int,
    max_edges: int,
    truncated: bool,
    truncation_reasons: tuple[GraphSliceTruncationReason, ...],
    nodes: tuple[EvidenceGraphNode, ...],
    edges: tuple[EvidenceGraphEdge, ...],
    node_depths: tuple[GraphNodeDepth, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:evidence-graph-slice:v1",
        {
            "graph_id": graph_id,
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "seed_entity_ids": seed_entity_ids,
            "traversal": traversal.value,
            "include_inferred": include_inferred,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "truncated": truncated,
            "truncation_reasons": [reason.value for reason in truncation_reasons],
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "edges": [edge.model_dump(mode="json") for edge in edges],
            "node_depths": [item.model_dump(mode="json") for item in node_depths],
        },
    )


def _canonical_hash(namespace: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()
