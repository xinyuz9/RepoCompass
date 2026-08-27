"""Adversarial contract tests for the content-addressed Evidence Graph."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.structure import (
    EvidenceGraph,
    EvidenceGraphConflictError,
    EvidenceNodeKind,
    EvidenceRelationKind,
    StructuralVerificationStatus,
    build_evidence_graph,
    build_python_file_index,
    create_evidence_graph,
    create_graph_edge,
    create_graph_node,
    merge_evidence_graphs,
)


REPOSITORY_ID = hashlib.sha256(b"evidence-graph-repository").hexdigest()
COMMIT_SHA = "b" * 40


def _graph(tmp_path: Path) -> EvidenceGraph:
    source = tmp_path / "pkg/service.py"
    source.parent.mkdir()
    source.write_text(
        "def helper():\n    return True\n\ndef run():\n    helper()\n    helper()\n",
        encoding="utf-8",
    )
    index = build_python_file_index(
        tmp_path,
        Path("pkg/service.py"),
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
    )
    return build_evidence_graph((index,))


def _reference_edge(graph: EvidenceGraph):
    return next(
        edge
        for edge in graph.edges
        if edge.relation is EvidenceRelationKind.REFERENCES
        and graph.node_by_entity_id(edge.target_entity_id).canonical_name == "helper"
    )


def test_graph_is_strict_frozen_content_addressed_and_json_round_trippable(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)

    restored = EvidenceGraph.model_validate_json(graph.model_dump_json())

    assert restored == graph
    assert restored.graph_id == graph.graph_id
    assert restored.repository_node.kind is EvidenceNodeKind.REPOSITORY
    with pytest.raises(ValidationError, match="frozen"):
        restored.commit_sha = "c" * 40  # type: ignore[misc]


@pytest.mark.parametrize("field", ["graph_id", "repository_id"])
def test_graph_rejects_top_level_identity_tampering(
    tmp_path: Path,
    field: str,
) -> None:
    graph = _graph(tmp_path)
    payload = graph.model_dump(mode="json")
    payload[field] = "f" * 64

    with pytest.raises(ValidationError, match="graph_id|repository|snapshot"):
        EvidenceGraph.model_validate_json(json.dumps(payload))


def test_graph_rejects_nested_node_edge_and_evidence_tampering(tmp_path: Path) -> None:
    graph = _graph(tmp_path)

    node_payload = graph.model_dump(mode="json")
    nodes = list(node_payload["nodes"])
    nodes[1] = {**nodes[1], "canonical_name": "tampered"}
    with pytest.raises(ValidationError, match="node_id|entity_id|identity"):
        EvidenceGraph.model_validate_json(json.dumps({**node_payload, "nodes": nodes}))

    edge_payload = graph.model_dump(mode="json")
    edges = list(edge_payload["edges"])
    edge = dict(edges[0])
    edge["edge_id"] = "f" * 64
    edges[0] = edge
    with pytest.raises(ValidationError, match="edge_id"):
        EvidenceGraph.model_validate_json(json.dumps({**edge_payload, "edges": edges}))

    evidence_payload = graph.model_dump(mode="json")
    edges = list(evidence_payload["edges"])
    edge = dict(next(item for item in edges if item["evidence"]))
    evidence = list(edge["evidence"])
    evidence[0] = {**evidence[0], "record_id": "f" * 64}
    edge["evidence"] = evidence
    edges[edges.index(next(item for item in edges if item["evidence"]))] = edge
    with pytest.raises(ValidationError, match="record_id|evidence_id"):
        EvidenceGraph.model_validate_json(
            json.dumps({**evidence_payload, "edges": edges})
        )


def test_graph_rejects_unknown_fields_coercion_versions_and_noncanonical_order(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    payload = graph.model_dump(mode="json")

    with pytest.raises(ValidationError, match="extra"):
        EvidenceGraph.model_validate_json(json.dumps({**payload, "secret": "x"}))
    with pytest.raises(ValidationError):
        EvidenceGraph.model_validate_json(
            json.dumps({**payload, "schema_version": "1"})
        )
    with pytest.raises(ValidationError):
        EvidenceGraph.model_validate_json(json.dumps({**payload, "schema_version": 2}))
    with pytest.raises(ValidationError, match="canonical"):
        EvidenceGraph.model_validate_json(
            json.dumps({**payload, "nodes": list(reversed(payload["nodes"]))})
        )


def test_verified_inferred_and_rejected_assertions_cannot_be_relabelled(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    verified = _reference_edge(graph)

    with pytest.raises(ValidationError, match="verified.*evidence"):
        create_graph_edge(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            relation=verified.relation,
            source_entity_id=verified.source_entity_id,
            target_entity_id=verified.target_entity_id,
            verification_status=StructuralVerificationStatus.VERIFIED,
        )
    with pytest.raises(ValidationError, match="explanation"):
        create_graph_edge(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            relation=verified.relation,
            source_entity_id=verified.source_entity_id,
            target_entity_id=verified.target_entity_id,
            verification_status=StructuralVerificationStatus.INFERRED,
        )
    with pytest.raises(ValidationError, match="explanation"):
        create_graph_edge(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            relation=verified.relation,
            source_entity_id=verified.source_entity_id,
            target_entity_id=verified.target_entity_id,
            verification_status=StructuralVerificationStatus.REJECTED,
        )
    with pytest.raises(ValidationError, match="verified.*explanation"):
        create_graph_edge(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            relation=verified.relation,
            source_entity_id=verified.source_entity_id,
            target_entity_id=verified.target_entity_id,
            verification_status=StructuralVerificationStatus.VERIFIED,
            evidence=verified.evidence,
            explanation="model guess",
        )

    inferred = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=verified.relation,
        source_entity_id=verified.source_entity_id,
        target_entity_id=verified.target_entity_id,
        verification_status=StructuralVerificationStatus.INFERRED,
        explanation="name similarity only",
    )
    file_node = next(node for node in graph.nodes if node.kind is EvidenceNodeKind.FILE)
    rejected = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=verified.relation,
        source_entity_id=file_node.entity_id,
        target_entity_id=verified.target_entity_id,
        verification_status=StructuralVerificationStatus.REJECTED,
        explanation="scope disproves the proposed relation",
    )
    inferred_graph = create_evidence_graph(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        nodes=graph.nodes,
        edges=(inferred, rejected),
    )

    assert inferred_graph.verified_edges == ()
    assert inferred_graph.accepted_edges == (inferred,)
    assert inferred_graph.rejected_edges == (rejected,)


def test_graph_rejects_dangling_self_incompatible_and_duplicate_assertions(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    edge = _reference_edge(graph)

    dangling = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=edge.relation,
        source_entity_id=edge.source_entity_id,
        target_entity_id="f" * 64,
        verification_status=StructuralVerificationStatus.INFERRED,
        explanation="candidate target",
    )
    with pytest.raises(ValidationError, match="dangling"):
        create_evidence_graph(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            nodes=graph.nodes,
            edges=(dangling,),
        )

    with pytest.raises(ValidationError, match="itself"):
        create_graph_edge(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            relation=edge.relation,
            source_entity_id=edge.source_entity_id,
            target_entity_id=edge.source_entity_id,
            verification_status=StructuralVerificationStatus.INFERRED,
            explanation="invalid self relation",
        )

    repository = graph.repository_node
    file_node = next(node for node in graph.nodes if node.kind is EvidenceNodeKind.FILE)
    incompatible = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=EvidenceRelationKind.IMPORTS,
        source_entity_id=file_node.entity_id,
        target_entity_id=repository.entity_id,
        verification_status=StructuralVerificationStatus.INFERRED,
        explanation="invalid target kind",
    )
    with pytest.raises(ValidationError, match="incompatible"):
        create_evidence_graph(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            nodes=graph.nodes,
            edges=(incompatible,),
        )

    inferred = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=edge.relation,
        source_entity_id=edge.source_entity_id,
        target_entity_id=edge.target_entity_id,
        verification_status=StructuralVerificationStatus.INFERRED,
        explanation="unverified duplicate",
    )
    with pytest.raises(ValidationError, match="assertion"):
        create_evidence_graph(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            nodes=graph.nodes,
            edges=(edge, inferred),
        )


def test_graph_rejects_real_evidence_attached_to_the_wrong_lexical_scope(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    edge = _reference_edge(graph)
    file_node = next(node for node in graph.nodes if node.kind is EvidenceNodeKind.FILE)
    wrong_scope = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=edge.relation,
        source_entity_id=file_node.entity_id,
        target_entity_id=edge.target_entity_id,
        verification_status=StructuralVerificationStatus.VERIFIED,
        evidence=edge.evidence,
    )

    with pytest.raises(ValidationError, match="source scope"):
        create_evidence_graph(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            nodes=graph.nodes,
            edges=(wrong_scope,),
        )


def test_merge_is_order_invariant_unions_provenance_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    edge = _reference_edge(graph)
    assert len(edge.evidence) >= 2
    left_edge = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=edge.relation,
        source_entity_id=edge.source_entity_id,
        target_entity_id=edge.target_entity_id,
        verification_status=edge.verification_status,
        evidence=(edge.evidence[0],),
    )
    right_edge = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=edge.relation,
        source_entity_id=edge.source_entity_id,
        target_entity_id=edge.target_entity_id,
        verification_status=edge.verification_status,
        evidence=(edge.evidence[1],),
    )
    left = create_evidence_graph(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        nodes=graph.nodes,
        edges=(left_edge,),
    )
    right = create_evidence_graph(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        nodes=graph.nodes,
        edges=(right_edge,),
    )

    merged = merge_evidence_graphs(left, right)

    assert merged == merge_evidence_graphs(right, left)
    assert len(merged.edges) == 1
    assert merged.edges[0].evidence == tuple(
        sorted(edge.evidence[:2], key=lambda item: item.evidence_id)
    )

    inferred = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=edge.relation,
        source_entity_id=edge.source_entity_id,
        target_entity_id=edge.target_entity_id,
        verification_status=StructuralVerificationStatus.INFERRED,
        explanation="not AST verified",
    )
    conflict = create_evidence_graph(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        nodes=graph.nodes,
        edges=(inferred,),
    )
    with pytest.raises(EvidenceGraphConflictError, match="status|assertion"):
        merge_evidence_graphs(left, conflict)

    other_repository_node = create_graph_node(
        repository_id=graph.repository_id,
        commit_sha="c" * 40,
        kind=EvidenceNodeKind.REPOSITORY,
        canonical_name="repository",
    )
    other_snapshot = create_evidence_graph(
        repository_id=graph.repository_id,
        commit_sha="c" * 40,
        nodes=(other_repository_node,),
    )
    with pytest.raises(
        EvidenceGraphConflictError,
        match="repository snapshots",
    ):
        merge_evidence_graphs(left, other_snapshot)
