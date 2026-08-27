"""Strict, content-addressed contracts for task-specific evidence graphs."""

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.structure.contracts import (
    PythonImportRecord,
    PythonReferenceRecord,
    PythonSymbolKind,
    PythonSymbolRecord,
    SourceLocation,
    StructuralVerificationStatus,
)
from repocompass.tools.contracts import CommitSha, Sha256Digest

EVIDENCE_GRAPH_SCHEMA_VERSION = 1
MAX_EVIDENCE_GRAPH_NODES = 50_000
MAX_EVIDENCE_GRAPH_EDGES = 100_000
MAX_EVIDENCE_PER_ASSERTION = 256
MAX_GRAPH_TEXT_CHARACTERS = 2_048

StructuralRecord: TypeAlias = (
    PythonSymbolRecord | PythonImportRecord | PythonReferenceRecord
)


class EvidenceNodeKind(StrEnum):
    """Closed node vocabulary supported by the first graph schema."""

    REPOSITORY = "repository"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    TEST = "test"
    EXTERNAL_MODULE = "external_module"
    UNRESOLVED_SYMBOL = "unresolved_symbol"


class EvidenceRelationKind(StrEnum):
    """Relations whose semantics can be validated by the graph boundary."""

    CONTAINS = "contains"
    IMPORTS = "imports"
    REFERENCES = "references"


class GraphEvidenceKind(StrEnum):
    """Structural record types admitted as graph provenance."""

    PYTHON_SYMBOL = "python_symbol"
    PYTHON_IMPORT = "python_import"
    PYTHON_REFERENCE = "python_reference"


class EvidenceGraphConflictError(ValueError):
    """Raised when deterministic merge would silently change a graph fact."""


class _EvidenceGraphModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class GraphEvidence(_EvidenceGraphModel):
    """Self-contained structural provenance for one node or relation."""

    schema_version: Literal[1] = EVIDENCE_GRAPH_SCHEMA_VERSION
    evidence_id: Sha256Digest
    kind: GraphEvidenceKind
    record_id: Sha256Digest
    location: SourceLocation
    record: StructuralRecord

    @model_validator(mode="after")
    def require_exact_content_addressed_record(self) -> "GraphEvidence":
        expected_kind, expected_id, expected_location = _record_identity(self.record)
        if self.kind is not expected_kind:
            raise ValueError("evidence kind does not match structural record")
        if self.record_id != expected_id:
            raise ValueError("record_id does not match structural record")
        if self.location != expected_location:
            raise ValueError("evidence location does not match structural record")
        expected_evidence_id = _evidence_id(
            kind=self.kind,
            record_id=self.record_id,
            location=self.location,
            record=self.record,
        )
        if self.evidence_id != expected_evidence_id:
            raise ValueError("evidence_id does not match graph evidence")
        return self


class EvidenceGraphNode(_EvidenceGraphModel):
    """One stable repository entity plus its verified structural provenance."""

    schema_version: Literal[1] = EVIDENCE_GRAPH_SCHEMA_VERSION
    node_id: Sha256Digest
    entity_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    kind: EvidenceNodeKind
    canonical_name: str = Field(min_length=1, max_length=MAX_GRAPH_TEXT_CHARACTERS)
    path: str | None = Field(default=None, min_length=1, max_length=4_096)
    symbol_id: Sha256Digest | None = None
    symbol_kind: PythonSymbolKind | None = None
    is_test: bool = False
    evidence: tuple[GraphEvidence, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_PER_ASSERTION,
    )

    @field_validator("canonical_name")
    @classmethod
    def require_safe_canonical_name(cls, value: str) -> str:
        _validate_graph_text(value, name="canonical_name")
        return value

    @field_validator("path")
    @classmethod
    def require_canonical_python_path(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_python_path(value)
        return value

    @model_validator(mode="after")
    def require_coherent_content_addressed_node(self) -> "EvidenceGraphNode":
        if self.evidence != canonical_graph_evidence(self.evidence):
            raise ValueError("node evidence must be unique and in canonical order")
        _require_unique_evidence(self.evidence)
        self._require_kind_fields()
        for item in self.evidence:
            if (
                item.location.repository_id != self.repository_id
                or item.location.commit_sha != self.commit_sha
            ):
                raise ValueError("node evidence has mismatched snapshot provenance")

        identity_payload = _node_identity_payload(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            kind=self.kind,
            canonical_name=self.canonical_name,
            path=self.path,
            symbol_id=self.symbol_id,
            symbol_kind=self.symbol_kind,
            is_test=self.is_test,
        )
        expected_entity_id = _canonical_hash(
            b"repocompass:evidence-entity:v1",
            identity_payload,
        )
        if self.entity_id != expected_entity_id:
            raise ValueError("entity_id does not match node identity")
        expected_node_id = _canonical_hash(
            b"repocompass:evidence-node:v1",
            {
                **identity_payload,
                "evidence": [item.model_dump(mode="json") for item in self.evidence],
            },
        )
        if self.node_id != expected_node_id:
            raise ValueError("node_id does not match node contents")
        return self

    def _require_kind_fields(self) -> None:
        placeholder_kinds = {
            EvidenceNodeKind.EXTERNAL_MODULE,
            EvidenceNodeKind.UNRESOLVED_SYMBOL,
        }
        structural_kinds = {
            EvidenceNodeKind.FILE,
            EvidenceNodeKind.CLASS,
            EvidenceNodeKind.FUNCTION,
            EvidenceNodeKind.METHOD,
            EvidenceNodeKind.TEST,
        }
        if self.kind is EvidenceNodeKind.REPOSITORY:
            if self.canonical_name != "repository":
                raise ValueError("repository node canonical_name must be repository")
            if (
                any(
                    value is not None
                    for value in (self.path, self.symbol_id, self.symbol_kind)
                )
                or self.is_test
                or self.evidence
            ):
                raise ValueError("repository node cannot carry source fields")
            return
        if self.kind in placeholder_kinds:
            if (
                any(
                    value is not None
                    for value in (self.path, self.symbol_id, self.symbol_kind)
                )
                or self.is_test
                or self.evidence
            ):
                raise ValueError("placeholder node cannot carry verified source fields")
            return
        if self.kind not in structural_kinds:
            raise ValueError("unsupported node kind")
        if self.path is None or self.symbol_id is None or self.symbol_kind is None:
            raise ValueError("structural node requires path and symbol identity")
        if len(self.evidence) != 1:
            raise ValueError("structural node requires one symbol evidence record")
        item = self.evidence[0]
        if item.kind is not GraphEvidenceKind.PYTHON_SYMBOL:
            raise ValueError("structural node requires Python symbol evidence")
        record = item.record
        if not isinstance(record, PythonSymbolRecord):
            raise ValueError("structural node evidence must contain a symbol record")
        if (
            self.symbol_id != record.symbol_id
            or self.symbol_kind is not record.kind
            or self.canonical_name != record.qualified_name
            or self.path != record.definition.path
        ):
            raise ValueError("structural node identity does not match symbol evidence")

        allowed_symbol_kinds = {
            EvidenceNodeKind.FILE: {PythonSymbolKind.MODULE},
            EvidenceNodeKind.CLASS: {PythonSymbolKind.CLASS},
            EvidenceNodeKind.FUNCTION: {
                PythonSymbolKind.FUNCTION,
                PythonSymbolKind.ASYNC_FUNCTION,
            },
            EvidenceNodeKind.METHOD: {
                PythonSymbolKind.METHOD,
                PythonSymbolKind.ASYNC_METHOD,
            },
            EvidenceNodeKind.TEST: {
                PythonSymbolKind.CLASS,
                PythonSymbolKind.FUNCTION,
                PythonSymbolKind.ASYNC_FUNCTION,
                PythonSymbolKind.METHOD,
                PythonSymbolKind.ASYNC_METHOD,
            },
        }
        if record.kind not in allowed_symbol_kinds[self.kind]:
            raise ValueError("node kind does not match Python symbol kind")
        if self.kind is EvidenceNodeKind.TEST:
            if not self.is_test or not record.is_test:
                raise ValueError("test node requires a test symbol")
        elif self.is_test:
            raise ValueError("only test nodes may set is_test")
        elif self.kind is not EvidenceNodeKind.FILE and record.is_test:
            raise ValueError("test symbols must use the test node kind")


class EvidenceGraphEdge(_EvidenceGraphModel):
    """One relation assertion with explicit verification status and lineage."""

    schema_version: Literal[1] = EVIDENCE_GRAPH_SCHEMA_VERSION
    edge_id: Sha256Digest
    assertion_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    relation: EvidenceRelationKind
    source_entity_id: Sha256Digest
    target_entity_id: Sha256Digest
    verification_status: StructuralVerificationStatus
    evidence: tuple[GraphEvidence, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_PER_ASSERTION,
    )
    explanation: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_GRAPH_TEXT_CHARACTERS,
    )

    @field_validator("explanation")
    @classmethod
    def require_safe_explanation(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_graph_text(value, name="explanation")
        return value

    @model_validator(mode="after")
    def require_coherent_content_addressed_assertion(self) -> "EvidenceGraphEdge":
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("graph relation cannot point to itself")
        if self.evidence != canonical_graph_evidence(self.evidence):
            raise ValueError("edge evidence must be unique and in canonical order")
        _require_unique_evidence(self.evidence)
        expected_kind = {
            EvidenceRelationKind.CONTAINS: GraphEvidenceKind.PYTHON_SYMBOL,
            EvidenceRelationKind.IMPORTS: GraphEvidenceKind.PYTHON_IMPORT,
            EvidenceRelationKind.REFERENCES: GraphEvidenceKind.PYTHON_REFERENCE,
        }[self.relation]
        if any(item.kind is not expected_kind for item in self.evidence):
            raise ValueError("relation evidence kind is incompatible")
        for item in self.evidence:
            if (
                item.location.repository_id != self.repository_id
                or item.location.commit_sha != self.commit_sha
            ):
                raise ValueError("edge evidence has mismatched snapshot provenance")

        if self.verification_status is StructuralVerificationStatus.VERIFIED:
            if not self.evidence:
                raise ValueError("verified relation requires structural evidence")
            if self.explanation is not None:
                raise ValueError("verified relation cannot carry an explanation")
        elif self.explanation is None:
            raise ValueError("inferred or rejected relation requires an explanation")

        assertion_payload = _assertion_identity_payload(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            relation=self.relation,
            source_entity_id=self.source_entity_id,
            target_entity_id=self.target_entity_id,
        )
        expected_assertion_id = _canonical_hash(
            b"repocompass:evidence-assertion:v1",
            assertion_payload,
        )
        if self.assertion_id != expected_assertion_id:
            raise ValueError("assertion_id does not match relation identity")
        expected_edge_id = _canonical_hash(
            b"repocompass:evidence-edge:v1",
            {
                **assertion_payload,
                "verification_status": self.verification_status.value,
                "evidence": [item.model_dump(mode="json") for item in self.evidence],
                "explanation": self.explanation,
            },
        )
        if self.edge_id != expected_edge_id:
            raise ValueError("edge_id does not match relation contents")
        return self


class EvidenceGraph(_EvidenceGraphModel):
    """Canonical graph for one immutable repository snapshot."""

    schema_version: Literal[1] = EVIDENCE_GRAPH_SCHEMA_VERSION
    graph_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    nodes: tuple[EvidenceGraphNode, ...] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_GRAPH_NODES,
    )
    edges: tuple[EvidenceGraphEdge, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_GRAPH_EDGES,
    )

    @model_validator(mode="after")
    def require_canonical_closed_content_addressed_graph(self) -> "EvidenceGraph":
        if self.nodes != canonical_graph_nodes(self.nodes):
            raise ValueError("graph nodes must be unique and in canonical order")
        if self.edges != canonical_graph_edges(self.edges):
            raise ValueError("graph edges must be unique and in canonical order")
        entity_ids = tuple(node.entity_id for node in self.nodes)
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("graph entity IDs must be unique")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node IDs must be unique")
        assertion_ids = tuple(edge.assertion_id for edge in self.edges)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("graph assertion IDs must be unique")
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("graph edge IDs must be unique")

        repository_nodes = [
            node for node in self.nodes if node.kind is EvidenceNodeKind.REPOSITORY
        ]
        if len(repository_nodes) != 1:
            raise ValueError("graph requires exactly one repository node")
        nodes_by_entity_id = {node.entity_id: node for node in self.nodes}
        for node in self.nodes:
            if (
                node.repository_id != self.repository_id
                or node.commit_sha != self.commit_sha
            ):
                raise ValueError("node has mismatched graph snapshot")
        for edge in self.edges:
            if (
                edge.repository_id != self.repository_id
                or edge.commit_sha != self.commit_sha
            ):
                raise ValueError("edge has mismatched graph snapshot")
            source = nodes_by_entity_id.get(edge.source_entity_id)
            target = nodes_by_entity_id.get(edge.target_entity_id)
            if source is None or target is None:
                raise ValueError("graph edge has a dangling endpoint")
            _validate_relation_compatibility(edge, source, target)
            _validate_relation_evidence(edge, source, target)

        expected_graph_id = _graph_id(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            nodes=self.nodes,
            edges=self.edges,
        )
        if self.graph_id != expected_graph_id:
            raise ValueError("graph_id does not match graph contents")
        return self

    @property
    def repository_node(self) -> EvidenceGraphNode:
        return next(
            node for node in self.nodes if node.kind is EvidenceNodeKind.REPOSITORY
        )

    @property
    def verified_edges(self) -> tuple[EvidenceGraphEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.verification_status is StructuralVerificationStatus.VERIFIED
        )

    @property
    def accepted_edges(self) -> tuple[EvidenceGraphEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.verification_status is not StructuralVerificationStatus.REJECTED
        )

    @property
    def rejected_edges(self) -> tuple[EvidenceGraphEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.verification_status is StructuralVerificationStatus.REJECTED
        )

    def node_by_entity_id(self, entity_id: str) -> EvidenceGraphNode:
        """Return an endpoint by stable identity without exposing mutable indexes."""

        for node in self.nodes:
            if node.entity_id == entity_id:
                return node
        raise KeyError(entity_id)


def create_graph_evidence(record: StructuralRecord) -> GraphEvidence:
    """Wrap one already-validated AST record in self-contained provenance."""

    if not isinstance(
        record,
        (PythonSymbolRecord, PythonImportRecord, PythonReferenceRecord),
    ):
        raise TypeError("record must be a supported Python structural record")
    kind, record_id, location = _record_identity(record)
    return GraphEvidence(
        evidence_id=_evidence_id(
            kind=kind,
            record_id=record_id,
            location=location,
            record=record,
        ),
        kind=kind,
        record_id=record_id,
        location=location,
        record=record,
    )


def create_graph_node(
    *,
    repository_id: str,
    commit_sha: str | None,
    kind: EvidenceNodeKind,
    canonical_name: str,
    path: str | None = None,
    symbol_id: str | None = None,
    symbol_kind: PythonSymbolKind | None = None,
    is_test: bool = False,
    evidence: tuple[GraphEvidence, ...] = (),
) -> EvidenceGraphNode:
    """Create one canonical node while keeping entity and record IDs distinct."""

    ordered_evidence = canonical_graph_evidence(evidence)
    identity_payload = _node_identity_payload(
        repository_id=repository_id,
        commit_sha=commit_sha,
        kind=kind,
        canonical_name=canonical_name,
        path=path,
        symbol_id=symbol_id,
        symbol_kind=symbol_kind,
        is_test=is_test,
    )
    entity_id = _canonical_hash(
        b"repocompass:evidence-entity:v1",
        identity_payload,
    )
    node_id = _canonical_hash(
        b"repocompass:evidence-node:v1",
        {
            **identity_payload,
            "evidence": [item.model_dump(mode="json") for item in ordered_evidence],
        },
    )
    return EvidenceGraphNode(
        node_id=node_id,
        entity_id=entity_id,
        repository_id=repository_id,
        commit_sha=commit_sha,
        kind=kind,
        canonical_name=canonical_name,
        path=path,
        symbol_id=symbol_id,
        symbol_kind=symbol_kind,
        is_test=is_test,
        evidence=ordered_evidence,
    )


def create_graph_edge(
    *,
    repository_id: str,
    commit_sha: str | None,
    relation: EvidenceRelationKind,
    source_entity_id: str,
    target_entity_id: str,
    verification_status: StructuralVerificationStatus,
    evidence: tuple[GraphEvidence, ...] = (),
    explanation: str | None = None,
) -> EvidenceGraphEdge:
    """Create one canonical assertion without promoting inference to fact."""

    ordered_evidence = canonical_graph_evidence(evidence)
    assertion_payload = _assertion_identity_payload(
        repository_id=repository_id,
        commit_sha=commit_sha,
        relation=relation,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
    )
    assertion_id = _canonical_hash(
        b"repocompass:evidence-assertion:v1",
        assertion_payload,
    )
    edge_id = _canonical_hash(
        b"repocompass:evidence-edge:v1",
        {
            **assertion_payload,
            "verification_status": verification_status.value,
            "evidence": [item.model_dump(mode="json") for item in ordered_evidence],
            "explanation": explanation,
        },
    )
    return EvidenceGraphEdge(
        edge_id=edge_id,
        assertion_id=assertion_id,
        repository_id=repository_id,
        commit_sha=commit_sha,
        relation=relation,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        verification_status=verification_status,
        evidence=ordered_evidence,
        explanation=explanation,
    )


def create_evidence_graph(
    *,
    repository_id: str,
    commit_sha: str | None,
    nodes: tuple[EvidenceGraphNode, ...],
    edges: tuple[EvidenceGraphEdge, ...] = (),
) -> EvidenceGraph:
    """Create a canonical graph and validate endpoint closure and semantics."""

    ordered_nodes = canonical_graph_nodes(nodes)
    ordered_edges = canonical_graph_edges(edges)
    return EvidenceGraph(
        graph_id=_graph_id(
            repository_id=repository_id,
            commit_sha=commit_sha,
            nodes=ordered_nodes,
            edges=ordered_edges,
        ),
        repository_id=repository_id,
        commit_sha=commit_sha,
        nodes=ordered_nodes,
        edges=ordered_edges,
    )


def merge_evidence_graphs(*graphs: EvidenceGraph) -> EvidenceGraph:
    """Merge equivalent snapshot graphs deterministically or fail on conflict."""

    if not graphs:
        raise ValueError("at least one evidence graph is required")
    if any(not isinstance(graph, EvidenceGraph) for graph in graphs):
        raise TypeError("all merge inputs must be EvidenceGraph instances")
    repository_id = graphs[0].repository_id
    commit_sha = graphs[0].commit_sha
    if any(
        graph.repository_id != repository_id or graph.commit_sha != commit_sha
        for graph in graphs[1:]
    ):
        raise EvidenceGraphConflictError("cannot merge different repository snapshots")

    nodes_by_entity: dict[str, EvidenceGraphNode] = {}
    for graph in graphs:
        for node in graph.nodes:
            previous = nodes_by_entity.get(node.entity_id)
            if previous is None:
                nodes_by_entity[node.entity_id] = node
                continue
            if _node_identity_tuple(previous) != _node_identity_tuple(node):
                raise EvidenceGraphConflictError("conflicting node identity")
            evidence = _merge_evidence(previous.evidence, node.evidence)
            nodes_by_entity[node.entity_id] = create_graph_node(
                repository_id=node.repository_id,
                commit_sha=node.commit_sha,
                kind=node.kind,
                canonical_name=node.canonical_name,
                path=node.path,
                symbol_id=node.symbol_id,
                symbol_kind=node.symbol_kind,
                is_test=node.is_test,
                evidence=evidence,
            )

    edges_by_assertion: dict[str, EvidenceGraphEdge] = {}
    for graph in graphs:
        for edge in graph.edges:
            previous = edges_by_assertion.get(edge.assertion_id)
            if previous is None:
                edges_by_assertion[edge.assertion_id] = edge
                continue
            if _assertion_identity_tuple(previous) != _assertion_identity_tuple(edge):
                raise EvidenceGraphConflictError("conflicting assertion identity")
            if previous.verification_status is not edge.verification_status:
                raise EvidenceGraphConflictError("conflicting assertion status")
            if previous.explanation != edge.explanation:
                raise EvidenceGraphConflictError("conflicting assertion explanation")
            evidence = _merge_evidence(previous.evidence, edge.evidence)
            edges_by_assertion[edge.assertion_id] = create_graph_edge(
                repository_id=edge.repository_id,
                commit_sha=edge.commit_sha,
                relation=edge.relation,
                source_entity_id=edge.source_entity_id,
                target_entity_id=edge.target_entity_id,
                verification_status=edge.verification_status,
                evidence=evidence,
                explanation=edge.explanation,
            )

    return create_evidence_graph(
        repository_id=repository_id,
        commit_sha=commit_sha,
        nodes=tuple(nodes_by_entity.values()),
        edges=tuple(edges_by_assertion.values()),
    )


def canonical_graph_evidence(
    evidence: tuple[GraphEvidence, ...],
) -> tuple[GraphEvidence, ...]:
    return tuple(sorted(evidence, key=lambda item: item.evidence_id))


def canonical_graph_nodes(
    nodes: tuple[EvidenceGraphNode, ...],
) -> tuple[EvidenceGraphNode, ...]:
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                node.kind.value,
                node.canonical_name,
                node.path or "",
                node.entity_id,
                node.node_id,
            ),
        )
    )


def canonical_graph_edges(
    edges: tuple[EvidenceGraphEdge, ...],
) -> tuple[EvidenceGraphEdge, ...]:
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.relation.value,
                edge.source_entity_id,
                edge.target_entity_id,
                edge.verification_status.value,
                edge.assertion_id,
                edge.edge_id,
            ),
        )
    )


def _record_identity(
    record: StructuralRecord,
) -> tuple[GraphEvidenceKind, str, SourceLocation]:
    if isinstance(record, PythonSymbolRecord):
        return GraphEvidenceKind.PYTHON_SYMBOL, record.symbol_id, record.definition
    if isinstance(record, PythonImportRecord):
        return GraphEvidenceKind.PYTHON_IMPORT, record.import_id, record.location
    if isinstance(record, PythonReferenceRecord):
        return GraphEvidenceKind.PYTHON_REFERENCE, record.reference_id, record.location
    raise TypeError("unsupported structural record")


def _evidence_id(
    *,
    kind: GraphEvidenceKind,
    record_id: str,
    location: SourceLocation,
    record: StructuralRecord,
) -> str:
    return _canonical_hash(
        b"repocompass:graph-evidence:v1",
        {
            "kind": kind.value,
            "record_id": record_id,
            "location": location.model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
        },
    )


def _node_identity_payload(
    *,
    repository_id: str,
    commit_sha: str | None,
    kind: EvidenceNodeKind,
    canonical_name: str,
    path: str | None,
    symbol_id: str | None,
    symbol_kind: PythonSymbolKind | None,
    is_test: bool,
) -> dict[str, object]:
    return {
        "repository_id": repository_id,
        "commit_sha": commit_sha,
        "kind": kind.value,
        "canonical_name": canonical_name,
        "path": path,
        "symbol_id": symbol_id,
        "symbol_kind": None if symbol_kind is None else symbol_kind.value,
        "is_test": is_test,
    }


def _assertion_identity_payload(
    *,
    repository_id: str,
    commit_sha: str | None,
    relation: EvidenceRelationKind,
    source_entity_id: str,
    target_entity_id: str,
) -> dict[str, object]:
    return {
        "repository_id": repository_id,
        "commit_sha": commit_sha,
        "relation": relation.value,
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
    }


def _graph_id(
    *,
    repository_id: str,
    commit_sha: str | None,
    nodes: tuple[EvidenceGraphNode, ...],
    edges: tuple[EvidenceGraphEdge, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:evidence-graph:v1",
        {
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "edges": [edge.model_dump(mode="json") for edge in edges],
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


def _require_unique_evidence(evidence: tuple[GraphEvidence, ...]) -> None:
    evidence_ids = tuple(item.evidence_id for item in evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence IDs must be unique")


def _merge_evidence(
    first: tuple[GraphEvidence, ...],
    second: tuple[GraphEvidence, ...],
) -> tuple[GraphEvidence, ...]:
    by_id: dict[str, GraphEvidence] = {}
    for item in (*first, *second):
        previous = by_id.get(item.evidence_id)
        if previous is not None and previous != item:
            raise EvidenceGraphConflictError("conflicting evidence record")
        by_id[item.evidence_id] = item
    if len(by_id) > MAX_EVIDENCE_PER_ASSERTION:
        raise EvidenceGraphConflictError("merged evidence exceeds assertion limit")
    return canonical_graph_evidence(tuple(by_id.values()))


def _node_identity_tuple(node: EvidenceGraphNode) -> tuple[object, ...]:
    return (
        node.repository_id,
        node.commit_sha,
        node.kind,
        node.canonical_name,
        node.path,
        node.symbol_id,
        node.symbol_kind,
        node.is_test,
    )


def _assertion_identity_tuple(edge: EvidenceGraphEdge) -> tuple[object, ...]:
    return (
        edge.repository_id,
        edge.commit_sha,
        edge.relation,
        edge.source_entity_id,
        edge.target_entity_id,
    )


def _validate_relation_compatibility(
    edge: EvidenceGraphEdge,
    source: EvidenceGraphNode,
    target: EvidenceGraphNode,
) -> None:
    structural = {
        EvidenceNodeKind.FILE,
        EvidenceNodeKind.CLASS,
        EvidenceNodeKind.FUNCTION,
        EvidenceNodeKind.METHOD,
        EvidenceNodeKind.TEST,
    }
    if edge.relation is EvidenceRelationKind.CONTAINS:
        valid = (
            source.kind is EvidenceNodeKind.REPOSITORY
            and target.kind is EvidenceNodeKind.FILE
        ) or (
            source.kind in structural
            and target.kind in structural - {EvidenceNodeKind.FILE}
        )
    elif edge.relation is EvidenceRelationKind.IMPORTS:
        valid = source.kind in structural and target.kind in {
            EvidenceNodeKind.FILE,
            EvidenceNodeKind.EXTERNAL_MODULE,
        }
    else:
        valid = source.kind in structural and target.kind in structural | {
            EvidenceNodeKind.UNRESOLVED_SYMBOL,
        }
    if not valid:
        raise ValueError("relation has incompatible endpoint node kinds")
    if (
        source.path is not None
        and target.path is not None
        and edge.relation is EvidenceRelationKind.CONTAINS
        and source.kind is not EvidenceNodeKind.REPOSITORY
        and source.path != target.path
    ):
        raise ValueError("contains relation cannot cross source files")


def _validate_relation_evidence(
    edge: EvidenceGraphEdge,
    source: EvidenceGraphNode,
    target: EvidenceGraphNode,
) -> None:
    for item in edge.evidence:
        if edge.relation is EvidenceRelationKind.CONTAINS:
            record = item.record
            if not isinstance(record, PythonSymbolRecord):
                raise ValueError("contains evidence must be a symbol record")
            if target.symbol_id != record.symbol_id:
                raise ValueError("contains evidence does not identify its target")
        elif edge.relation is EvidenceRelationKind.IMPORTS:
            record = item.record
            if not isinstance(record, PythonImportRecord):
                raise ValueError("imports evidence must be an import record")
            if source.symbol_id != record.scope_symbol_id:
                raise ValueError("imports evidence does not match source scope")
        else:
            record = item.record
            if not isinstance(record, PythonReferenceRecord):
                raise ValueError("references evidence must be a reference record")
            if source.symbol_id != record.scope_symbol_id:
                raise ValueError("references evidence does not match source scope")
        if source.path is not None and item.location.path != source.path:
            raise ValueError("relation evidence path does not match source node")


def _validate_graph_text(value: str, *, name: str) -> None:
    if (
        value.strip() != value
        or "\x00" in value
        or any(character in value for character in ("\r", "\n"))
    ):
        raise ValueError(f"{name} contains unsafe or non-canonical text")


def _validate_python_path(value: str) -> None:
    if "\\" in value or "\x00" in value:
        raise ValueError("path must be a normalized POSIX Python path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value == "."
        or ".." in path.parts
        or path.as_posix() != value
        or path.suffix != ".py"
    ):
        raise ValueError("path must be a normalized relative .py path")
