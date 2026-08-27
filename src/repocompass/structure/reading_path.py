"""Traceable reading steps and claim lineage over bounded graph slices."""

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.structure.contracts import (
    SourceLocation,
    StructuralVerificationStatus,
)
from repocompass.structure.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    EvidenceNodeKind,
    EvidenceRelationKind,
    GraphEvidence,
    canonical_graph_evidence,
)
from repocompass.structure.graph_slice import (
    EvidenceGraphSlice,
    GraphTraversalDirection,
    ReadingPathError,
    ReadingPathErrorReason,
    slice_evidence_graph,
    traversal_predecessor_candidates,
)
from repocompass.tools.contracts import Sha256Digest

READING_PATH_SCHEMA_VERSION = 1
MAX_READING_PATH_CLAIMS = 64
MAX_CLAIM_ASSERTIONS = 64
MAX_LINEAGE_EVIDENCE = 4_096
MAX_READING_TEXT_CHARACTERS = 8_192


class ReadingStepKind(StrEnum):
    """Source-backed entities that can appear in an ordered reading path."""

    FILE = "file"
    SYMBOL = "symbol"
    TEST = "test"


class ReadingStepReason(StrEnum):
    """Stable reason why a source-backed entity was selected."""

    SEED = "seed"
    STRUCTURAL_CONTEXT = "structural_context"
    IMPORT_DEPENDENCY = "import_dependency"
    REFERENCE = "reference"
    RELATED_TEST = "related_test"


class _ReadingPathModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class TraceableClaim(_ReadingPathModel):
    """A public claim explicitly bound to graph assertion identities."""

    schema_version: Literal[1] = READING_PATH_SCHEMA_VERSION
    claim_id: Sha256Digest
    text: str = Field(min_length=1, max_length=MAX_READING_TEXT_CHARACTERS)
    supporting_assertion_ids: tuple[Sha256Digest, ...] = Field(
        min_length=1,
        max_length=MAX_CLAIM_ASSERTIONS,
    )
    explanation: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_READING_TEXT_CHARACTERS,
    )

    @field_validator("text", "explanation")
    @classmethod
    def require_safe_claim_text(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_text(value, name="claim text")
        return value

    @model_validator(mode="after")
    def require_content_addressed_canonical_claim(self) -> "TraceableClaim":
        if self.supporting_assertion_ids != tuple(
            sorted(self.supporting_assertion_ids)
        ) or len(self.supporting_assertion_ids) != len(
            set(self.supporting_assertion_ids)
        ):
            raise ValueError("claim assertion IDs must be unique and canonical")
        expected = _claim_id(
            text=self.text,
            supporting_assertion_ids=self.supporting_assertion_ids,
            explanation=self.explanation,
        )
        if self.claim_id != expected:
            raise ValueError("claim_id does not match claim contents")
        return self


class ClaimLineage(_ReadingPathModel):
    """A claim linked through accepted assertions to exact source provenance."""

    schema_version: Literal[1] = READING_PATH_SCHEMA_VERSION
    lineage_id: Sha256Digest
    claim_id: Sha256Digest
    text: str = Field(min_length=1, max_length=MAX_READING_TEXT_CHARACTERS)
    verification_status: StructuralVerificationStatus
    supporting_assertion_ids: tuple[Sha256Digest, ...] = Field(
        min_length=1,
        max_length=MAX_CLAIM_ASSERTIONS,
    )
    evidence: tuple[GraphEvidence, ...] = Field(
        default=(),
        max_length=MAX_LINEAGE_EVIDENCE,
    )
    source_locations: tuple[SourceLocation, ...] = Field(
        default=(),
        max_length=MAX_LINEAGE_EVIDENCE,
    )
    provenance_complete: bool
    explanation: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_READING_TEXT_CHARACTERS,
    )

    @field_validator("text", "explanation")
    @classmethod
    def require_safe_lineage_text(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_text(value, name="lineage text")
        return value

    @model_validator(mode="after")
    def require_honest_content_addressed_lineage(self) -> "ClaimLineage":
        if self.verification_status is StructuralVerificationStatus.REJECTED:
            raise ValueError("rejected assertions cannot form claim lineage")
        if self.supporting_assertion_ids != tuple(
            sorted(self.supporting_assertion_ids)
        ) or len(self.supporting_assertion_ids) != len(
            set(self.supporting_assertion_ids)
        ):
            raise ValueError("lineage assertion IDs must be unique and canonical")
        if self.evidence != canonical_graph_evidence(self.evidence):
            raise ValueError("lineage evidence must be unique and canonical")
        _require_unique_ids(
            tuple(item.evidence_id for item in self.evidence),
            name="lineage evidence",
        )
        if self.source_locations != _canonical_locations(self.source_locations):
            raise ValueError("lineage source locations must be unique and canonical")
        _require_unique_ids(
            tuple(item.location_id for item in self.source_locations),
            name="lineage source location",
        )
        if self.verification_status is StructuralVerificationStatus.VERIFIED:
            if (
                not self.provenance_complete
                or not self.evidence
                or not self.source_locations
            ):
                raise ValueError("verified claim requires complete source provenance")
            if self.explanation is not None:
                raise ValueError("verified claim cannot carry an inference explanation")
        else:
            if self.provenance_complete:
                raise ValueError("inferred claim cannot claim complete provenance")
            if self.explanation is None:
                raise ValueError("inferred claim requires an explanation")
        expected_claim_id = _claim_id(
            text=self.text,
            supporting_assertion_ids=self.supporting_assertion_ids,
            explanation=self.explanation,
        )
        if self.claim_id != expected_claim_id:
            raise ValueError("claim_id does not match lineage claim contents")
        expected_lineage_id = _lineage_id(
            claim_id=self.claim_id,
            text=self.text,
            verification_status=self.verification_status,
            supporting_assertion_ids=self.supporting_assertion_ids,
            evidence=self.evidence,
            source_locations=self.source_locations,
            provenance_complete=self.provenance_complete,
            explanation=self.explanation,
        )
        if self.lineage_id != expected_lineage_id:
            raise ValueError("lineage_id does not match claim lineage")
        return self


class ReadingStep(_ReadingPathModel):
    """One deterministic source location in a task-specific reading order."""

    schema_version: Literal[1] = READING_PATH_SCHEMA_VERSION
    reading_step_id: Sha256Digest
    ordinal: int = Field(ge=1, le=2_000)
    entity_id: Sha256Digest
    node_id: Sha256Digest
    kind: ReadingStepKind
    reason: ReadingStepReason
    canonical_name: str = Field(min_length=1, max_length=2_048)
    path: str = Field(min_length=1, max_length=4_096)
    depth: int = Field(ge=0, le=8)
    path_verification_status: StructuralVerificationStatus
    via_assertion_id: Sha256Digest | None = None
    source_locations: tuple[SourceLocation, ...] = Field(
        min_length=1,
        max_length=256,
    )

    @field_validator("canonical_name", "path")
    @classmethod
    def require_safe_step_text(cls, value: str) -> str:
        _validate_text(value, name="reading step text")
        return value

    @model_validator(mode="after")
    def require_content_addressed_step(self) -> "ReadingStep":
        if self.path_verification_status is StructuralVerificationStatus.REJECTED:
            raise ValueError("reading path cannot use a rejected relation")
        if self.source_locations != _canonical_locations(self.source_locations):
            raise ValueError("step source locations must be unique and canonical")
        _require_unique_ids(
            tuple(item.location_id for item in self.source_locations),
            name="step source location",
        )
        if any(item.path != self.path for item in self.source_locations):
            raise ValueError("step source location path does not match step path")
        if self.depth == 0:
            if (
                self.via_assertion_id is not None
                or self.reason is not ReadingStepReason.SEED
            ):
                raise ValueError(
                    "seed reading step cannot have a predecessor assertion"
                )
        elif self.via_assertion_id is None or self.reason is ReadingStepReason.SEED:
            raise ValueError("non-seed reading step requires a predecessor assertion")
        expected = _reading_step_id(
            ordinal=self.ordinal,
            entity_id=self.entity_id,
            node_id=self.node_id,
            kind=self.kind,
            reason=self.reason,
            canonical_name=self.canonical_name,
            path=self.path,
            depth=self.depth,
            path_verification_status=self.path_verification_status,
            via_assertion_id=self.via_assertion_id,
            source_locations=self.source_locations,
        )
        if self.reading_step_id != expected:
            raise ValueError("reading_step_id does not match reading step contents")
        return self


class TraceableReadingPath(_ReadingPathModel):
    """Stable UI/MCP artifact combining a bounded graph, steps, and claims."""

    schema_version: Literal[1] = READING_PATH_SCHEMA_VERSION
    reading_path_id: Sha256Digest
    objective: str = Field(min_length=1, max_length=MAX_READING_TEXT_CHARACTERS)
    graph_slice: EvidenceGraphSlice
    steps: tuple[ReadingStep, ...] = Field(min_length=1, max_length=2_000)
    claim_lineages: tuple[ClaimLineage, ...] = Field(
        default=(),
        max_length=MAX_READING_PATH_CLAIMS,
    )

    @field_validator("objective")
    @classmethod
    def require_safe_objective(cls, value: str) -> str:
        _validate_text(value, name="objective")
        return value

    @model_validator(mode="after")
    def require_complete_canonical_content_addressed_path(
        self,
    ) -> "TraceableReadingPath":
        if tuple(step.ordinal for step in self.steps) != tuple(
            range(1, len(self.steps) + 1)
        ):
            raise ValueError("reading step ordinals must be contiguous and canonical")
        _require_unique_ids(
            tuple(step.reading_step_id for step in self.steps),
            name="reading step",
        )
        _require_unique_ids(
            tuple(step.entity_id for step in self.steps),
            name="reading step entity",
        )
        if self.claim_lineages != tuple(
            sorted(self.claim_lineages, key=lambda item: item.claim_id)
        ):
            raise ValueError("claim lineages must be unique and canonical")
        _require_unique_ids(
            tuple(item.claim_id for item in self.claim_lineages),
            name="claim lineage",
        )

        edges_by_id = {edge.assertion_id: edge for edge in self.graph_slice.edges}
        metadata = _derive_path_metadata(self.graph_slice)
        expected_nodes = tuple(
            sorted(
                (node for node in self.graph_slice.nodes if _is_source_node(node)),
                key=_reading_node_sort_key,
            )
        )
        if tuple(step.entity_id for step in self.steps) != tuple(
            node.entity_id for node in expected_nodes
        ):
            raise ValueError("reading steps are incomplete or not in canonical order")
        for step, node in zip(self.steps, expected_nodes, strict=True):
            via_edge, path_status = metadata[node.entity_id]
            expected_locations = _node_source_locations(node)
            if (
                step.node_id != node.node_id
                or step.canonical_name != node.canonical_name
                or step.path != node.path
                or step.depth != self.graph_slice.depth_for(node.entity_id)
                or step.kind is not _reading_step_kind(node)
                or step.source_locations != expected_locations
                or step.path_verification_status is not path_status
                or step.via_assertion_id
                != (None if via_edge is None else via_edge.assertion_id)
                or step.reason is not _reading_step_reason(node, via_edge)
            ):
                raise ValueError("reading step does not match graph slice lineage")
            if (
                step.via_assertion_id is not None
                and step.via_assertion_id not in edges_by_id
            ):
                raise ValueError("reading step predecessor is outside graph slice")

        for lineage in self.claim_lineages:
            lineage_edges = tuple(
                edges_by_id.get(assertion_id)
                for assertion_id in lineage.supporting_assertion_ids
            )
            if any(edge is None for edge in lineage_edges):
                raise ValueError("claim lineage assertion is outside graph slice")
            expected_status, evidence, locations = _lineage_material(
                tuple(edge for edge in lineage_edges if edge is not None)
            )
            if (
                lineage.verification_status is not expected_status
                or lineage.evidence != evidence
                or lineage.source_locations != locations
                or lineage.provenance_complete
                is not (expected_status is StructuralVerificationStatus.VERIFIED)
            ):
                raise ValueError("claim lineage does not match graph assertions")

        expected_path_id = _reading_path_id(
            objective=self.objective,
            graph_slice=self.graph_slice,
            steps=self.steps,
            claim_lineages=self.claim_lineages,
        )
        if self.reading_path_id != expected_path_id:
            raise ValueError("reading_path_id does not match reading path contents")
        return self


def create_traceable_claim(
    *,
    text: str,
    supporting_assertion_ids: tuple[str, ...],
    explanation: str | None = None,
) -> TraceableClaim:
    """Create a canonical claim without assigning its verification status."""

    ordered_ids = tuple(sorted(supporting_assertion_ids))
    return TraceableClaim(
        claim_id=_claim_id(
            text=text,
            supporting_assertion_ids=ordered_ids,
            explanation=explanation,
        ),
        text=text,
        supporting_assertion_ids=ordered_ids,
        explanation=explanation,
    )


def build_traceable_reading_path(
    graph: EvidenceGraph,
    *,
    objective: str,
    seed_entity_ids: tuple[str, ...],
    claims: tuple[TraceableClaim, ...] = (),
    traversal: GraphTraversalDirection = GraphTraversalDirection.BOTH,
    include_inferred: bool = False,
    max_depth: int = 3,
    max_nodes: int = 100,
    max_edges: int = 200,
) -> TraceableReadingPath:
    """Build a deterministic source reading path and auditable claim lineage."""

    if not isinstance(claims, tuple) or any(
        not isinstance(claim, TraceableClaim) for claim in claims
    ):
        raise TypeError("claims must be a tuple of TraceableClaim instances")
    _validate_text(objective, name="objective")
    graph_slice = slice_evidence_graph(
        graph,
        seed_entity_ids=seed_entity_ids,
        traversal=traversal,
        include_inferred=include_inferred,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    metadata = _derive_path_metadata(graph_slice)
    source_nodes = tuple(
        sorted(
            (node for node in graph_slice.nodes if _is_source_node(node)),
            key=_reading_node_sort_key,
        )
    )
    if not source_nodes:
        raise ReadingPathError(ReadingPathErrorReason.NO_SOURCE_STEPS)
    steps = tuple(
        _create_reading_step(
            ordinal=ordinal,
            node=node,
            depth=graph_slice.depth_for(node.entity_id),
            via_edge=metadata[node.entity_id][0],
            path_status=metadata[node.entity_id][1],
        )
        for ordinal, node in enumerate(source_nodes, start=1)
    )
    ordered_claims = tuple(sorted(claims, key=lambda claim: claim.claim_id))
    if len(ordered_claims) != len({claim.claim_id for claim in ordered_claims}):
        raise ValueError("claim IDs must be unique")
    lineages = tuple(
        _build_claim_lineage(claim, graph=graph, graph_slice=graph_slice)
        for claim in ordered_claims
    )
    return _create_reading_path(
        objective=objective,
        graph_slice=graph_slice,
        steps=steps,
        claim_lineages=lineages,
    )


def _build_claim_lineage(
    claim: TraceableClaim,
    *,
    graph: EvidenceGraph,
    graph_slice: EvidenceGraphSlice,
) -> ClaimLineage:
    graph_edges = {edge.assertion_id: edge for edge in graph.edges}
    slice_edges = {edge.assertion_id: edge for edge in graph_slice.edges}
    selected: list[EvidenceGraphEdge] = []
    for assertion_id in claim.supporting_assertion_ids:
        graph_edge = graph_edges.get(assertion_id)
        if graph_edge is None:
            raise ReadingPathError(ReadingPathErrorReason.UNKNOWN_ASSERTION)
        if graph_edge.verification_status is StructuralVerificationStatus.REJECTED:
            raise ReadingPathError(ReadingPathErrorReason.REJECTED_ASSERTION)
        edge = slice_edges.get(assertion_id)
        if edge is None:
            raise ReadingPathError(ReadingPathErrorReason.ASSERTION_OUTSIDE_SLICE)
        selected.append(edge)
    status, evidence, locations = _lineage_material(tuple(selected))
    if status is StructuralVerificationStatus.INFERRED and claim.explanation is None:
        raise ReadingPathError(
            ReadingPathErrorReason.INFERRED_CLAIM_EXPLANATION_REQUIRED
        )
    if (
        status is StructuralVerificationStatus.VERIFIED
        and claim.explanation is not None
    ):
        raise ReadingPathError(ReadingPathErrorReason.VERIFIED_CLAIM_EXPLANATION)
    provenance_complete = status is StructuralVerificationStatus.VERIFIED
    lineage_id = _lineage_id(
        claim_id=claim.claim_id,
        text=claim.text,
        verification_status=status,
        supporting_assertion_ids=claim.supporting_assertion_ids,
        evidence=evidence,
        source_locations=locations,
        provenance_complete=provenance_complete,
        explanation=claim.explanation,
    )
    return ClaimLineage(
        lineage_id=lineage_id,
        claim_id=claim.claim_id,
        text=claim.text,
        verification_status=status,
        supporting_assertion_ids=claim.supporting_assertion_ids,
        evidence=evidence,
        source_locations=locations,
        provenance_complete=provenance_complete,
        explanation=claim.explanation,
    )


def _lineage_material(
    edges: tuple[EvidenceGraphEdge, ...],
) -> tuple[
    StructuralVerificationStatus,
    tuple[GraphEvidence, ...],
    tuple[SourceLocation, ...],
]:
    status = (
        StructuralVerificationStatus.VERIFIED
        if all(
            edge.verification_status is StructuralVerificationStatus.VERIFIED
            for edge in edges
        )
        else StructuralVerificationStatus.INFERRED
    )
    evidence_by_id = {
        item.evidence_id: item for edge in edges for item in edge.evidence
    }
    evidence = canonical_graph_evidence(tuple(evidence_by_id.values()))
    locations = _canonical_locations(
        tuple({item.location.location_id: item.location for item in evidence}.values())
    )
    return status, evidence, locations


def _derive_path_metadata(
    graph_slice: EvidenceGraphSlice,
) -> dict[
    str,
    tuple[EvidenceGraphEdge | None, StructuralVerificationStatus],
]:
    metadata: dict[
        str,
        tuple[EvidenceGraphEdge | None, StructuralVerificationStatus],
    ] = {
        entity_id: (None, StructuralVerificationStatus.VERIFIED)
        for entity_id in graph_slice.seed_entity_ids
    }
    for depth in range(1, graph_slice.max_depth + 1):
        entity_ids = sorted(
            item.entity_id for item in graph_slice.node_depths if item.depth == depth
        )
        for entity_id in entity_ids:
            candidates = traversal_predecessor_candidates(graph_slice, entity_id)
            if not candidates:
                raise ValueError("slice node has no depth-decreasing predecessor")
            edge, predecessor_id = sorted(
                candidates,
                key=lambda item: (
                    0
                    if item[0].verification_status
                    is StructuralVerificationStatus.VERIFIED
                    else 1,
                    item[0].relation.value,
                    item[1],
                    item[0].assertion_id,
                ),
            )[0]
            predecessor_status = metadata[predecessor_id][1]
            status = (
                StructuralVerificationStatus.VERIFIED
                if predecessor_status is StructuralVerificationStatus.VERIFIED
                and edge.verification_status is StructuralVerificationStatus.VERIFIED
                else StructuralVerificationStatus.INFERRED
            )
            metadata[entity_id] = (edge, status)
    return metadata


def _create_reading_step(
    *,
    ordinal: int,
    node: EvidenceGraphNode,
    depth: int,
    via_edge: EvidenceGraphEdge | None,
    path_status: StructuralVerificationStatus,
) -> ReadingStep:
    locations = _node_source_locations(node)
    payload = {
        "ordinal": ordinal,
        "entity_id": node.entity_id,
        "node_id": node.node_id,
        "kind": _reading_step_kind(node),
        "reason": _reading_step_reason(node, via_edge),
        "canonical_name": node.canonical_name,
        "path": node.path,
        "depth": depth,
        "path_verification_status": path_status,
        "via_assertion_id": None if via_edge is None else via_edge.assertion_id,
        "source_locations": locations,
    }
    if payload["path"] is None:
        raise ValueError("source-backed reading step requires a path")
    return ReadingStep(
        reading_step_id=_reading_step_id(**payload),
        **payload,
    )


def _create_reading_path(
    *,
    objective: str,
    graph_slice: EvidenceGraphSlice,
    steps: tuple[ReadingStep, ...],
    claim_lineages: tuple[ClaimLineage, ...],
) -> TraceableReadingPath:
    return TraceableReadingPath(
        reading_path_id=_reading_path_id(
            objective=objective,
            graph_slice=graph_slice,
            steps=steps,
            claim_lineages=claim_lineages,
        ),
        objective=objective,
        graph_slice=graph_slice,
        steps=steps,
        claim_lineages=claim_lineages,
    )


def _is_source_node(node: EvidenceGraphNode) -> bool:
    return node.kind in {
        EvidenceNodeKind.FILE,
        EvidenceNodeKind.CLASS,
        EvidenceNodeKind.FUNCTION,
        EvidenceNodeKind.METHOD,
        EvidenceNodeKind.TEST,
    }


def _reading_step_kind(node: EvidenceGraphNode) -> ReadingStepKind:
    if node.kind is EvidenceNodeKind.FILE:
        return ReadingStepKind.FILE
    if node.kind is EvidenceNodeKind.TEST:
        return ReadingStepKind.TEST
    return ReadingStepKind.SYMBOL


def _reading_step_reason(
    node: EvidenceGraphNode,
    via_edge: EvidenceGraphEdge | None,
) -> ReadingStepReason:
    if via_edge is None:
        return ReadingStepReason.SEED
    if node.kind is EvidenceNodeKind.TEST:
        return ReadingStepReason.RELATED_TEST
    return {
        EvidenceRelationKind.CONTAINS: ReadingStepReason.STRUCTURAL_CONTEXT,
        EvidenceRelationKind.IMPORTS: ReadingStepReason.IMPORT_DEPENDENCY,
        EvidenceRelationKind.REFERENCES: ReadingStepReason.REFERENCE,
    }[via_edge.relation]


def _reading_node_sort_key(node: EvidenceGraphNode) -> tuple[object, ...]:
    kind_rank = {
        EvidenceNodeKind.FILE: 0,
        EvidenceNodeKind.CLASS: 1,
        EvidenceNodeKind.FUNCTION: 1,
        EvidenceNodeKind.METHOD: 1,
        EvidenceNodeKind.TEST: 2,
    }[node.kind]
    location = _node_source_locations(node)[0]
    return (
        kind_rank,
        node.path or "",
        location.line_start,
        location.column_start,
        node.canonical_name,
        node.entity_id,
    )


def _node_source_locations(
    node: EvidenceGraphNode,
) -> tuple[SourceLocation, ...]:
    return _canonical_locations(tuple(item.location for item in node.evidence))


def _canonical_locations(
    locations: tuple[SourceLocation, ...],
) -> tuple[SourceLocation, ...]:
    return tuple(
        sorted(
            locations,
            key=lambda item: (
                item.path,
                item.line_start,
                item.column_start,
                item.line_end,
                item.column_end,
                item.location_id,
            ),
        )
    )


def _claim_id(
    *,
    text: str,
    supporting_assertion_ids: tuple[str, ...],
    explanation: str | None,
) -> str:
    return _canonical_hash(
        b"repocompass:traceable-claim:v1",
        {
            "text": text,
            "supporting_assertion_ids": supporting_assertion_ids,
            "explanation": explanation,
        },
    )


def _lineage_id(
    *,
    claim_id: str,
    text: str,
    verification_status: StructuralVerificationStatus,
    supporting_assertion_ids: tuple[str, ...],
    evidence: tuple[GraphEvidence, ...],
    source_locations: tuple[SourceLocation, ...],
    provenance_complete: bool,
    explanation: str | None,
) -> str:
    return _canonical_hash(
        b"repocompass:claim-lineage:v1",
        {
            "claim_id": claim_id,
            "text": text,
            "verification_status": verification_status.value,
            "supporting_assertion_ids": supporting_assertion_ids,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "source_locations": [
                item.model_dump(mode="json") for item in source_locations
            ],
            "provenance_complete": provenance_complete,
            "explanation": explanation,
        },
    )


def _reading_step_id(
    *,
    ordinal: int,
    entity_id: str,
    node_id: str,
    kind: ReadingStepKind,
    reason: ReadingStepReason,
    canonical_name: str,
    path: str,
    depth: int,
    path_verification_status: StructuralVerificationStatus,
    via_assertion_id: str | None,
    source_locations: tuple[SourceLocation, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:reading-step:v1",
        {
            "ordinal": ordinal,
            "entity_id": entity_id,
            "node_id": node_id,
            "kind": kind.value,
            "reason": reason.value,
            "canonical_name": canonical_name,
            "path": path,
            "depth": depth,
            "path_verification_status": path_verification_status.value,
            "via_assertion_id": via_assertion_id,
            "source_locations": [
                item.model_dump(mode="json") for item in source_locations
            ],
        },
    )


def _reading_path_id(
    *,
    objective: str,
    graph_slice: EvidenceGraphSlice,
    steps: tuple[ReadingStep, ...],
    claim_lineages: tuple[ClaimLineage, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:traceable-reading-path:v1",
        {
            "objective": objective,
            "graph_slice": graph_slice.model_dump(mode="json"),
            "steps": [step.model_dump(mode="json") for step in steps],
            "claim_lineages": [
                lineage.model_dump(mode="json") for lineage in claim_lineages
            ],
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


def _require_unique_ids(values: tuple[str, ...], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} IDs must be unique")


def _validate_text(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_READING_TEXT_CHARACTERS
        or value.strip() != value
        or "\x00" in value
        or any(character in value for character in ("\r", "\n"))
    ):
        raise ValueError(f"{name} contains unsafe or non-canonical text")
