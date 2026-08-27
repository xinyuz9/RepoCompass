"""Deterministic multi-file AST-to-Evidence-Graph tests."""

import hashlib
from pathlib import Path

import pytest

from repocompass.structure import (
    EvidenceGraph,
    EvidenceNodeKind,
    EvidenceRelationKind,
    GraphEvidenceKind,
    StructuralVerificationStatus,
    build_evidence_graph,
    build_python_file_index,
)


REPOSITORY_ID = hashlib.sha256(b"graph-builder-repository").hexdigest()
COMMIT_SHA = "d" * 40


def _indexes(tmp_path: Path):
    sources = {
        "pkg/core.py": (
            "def helper():\n"
            "    return True\n\n"
            "class Service:\n"
            "    def run(self):\n"
            "        helper()\n"
            "        helper()\n"
            "        return True\n"
        ),
        "pkg/api.py": (
            "from .core import Service\n"
            "import json\n\n"
            "def build():\n"
            "    return Service().run()\n"
        ),
        "tests/test_api.py": (
            "from pkg.api import build\n\ndef test_build():\n    assert build()\n"
        ),
    }
    indexes = []
    for relative, source in sources.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        indexes.append(
            build_python_file_index(
                tmp_path,
                Path(relative),
                repository_id=REPOSITORY_ID,
                commit_sha=COMMIT_SHA,
            )
        )
    return tuple(indexes)


def _node(graph: EvidenceGraph, name: str, kind: EvidenceNodeKind):
    return next(
        node
        for node in graph.nodes
        if node.canonical_name == name and node.kind is kind
    )


def test_builder_creates_conservative_typed_graph_with_exact_provenance(
    tmp_path: Path,
) -> None:
    indexes = _indexes(tmp_path)

    graph = build_evidence_graph(indexes)

    assert graph == build_evidence_graph(tuple(reversed(indexes)))
    assert graph.repository_id == REPOSITORY_ID
    assert graph.commit_sha == COMMIT_SHA
    assert (
        len([node for node in graph.nodes if node.kind is EvidenceNodeKind.FILE]) == 3
    )
    assert _node(graph, "pkg.core.Service", EvidenceNodeKind.CLASS)
    assert _node(graph, "pkg.core.Service.run", EvidenceNodeKind.METHOD)
    assert _node(graph, "tests.test_api.test_build", EvidenceNodeKind.TEST)

    assert {edge.relation for edge in graph.edges} == {
        EvidenceRelationKind.CONTAINS,
        EvidenceRelationKind.IMPORTS,
        EvidenceRelationKind.REFERENCES,
    }
    assert all(
        edge.verification_status is StructuralVerificationStatus.VERIFIED
        for edge in graph.edges
    )
    assert all(edge.evidence for edge in graph.edges)
    for edge in graph.edges:
        expected_kind = {
            EvidenceRelationKind.CONTAINS: GraphEvidenceKind.PYTHON_SYMBOL,
            EvidenceRelationKind.IMPORTS: GraphEvidenceKind.PYTHON_IMPORT,
            EvidenceRelationKind.REFERENCES: GraphEvidenceKind.PYTHON_REFERENCE,
        }[edge.relation]
        assert {item.kind for item in edge.evidence} == {expected_kind}
        assert all(
            item.location.repository_id == REPOSITORY_ID for item in edge.evidence
        )
        assert all(item.location.commit_sha == COMMIT_SHA for item in edge.evidence)


def test_builder_resolves_only_exact_internal_modules_and_keeps_references_unresolved(
    tmp_path: Path,
) -> None:
    graph = build_evidence_graph(_indexes(tmp_path))
    api_file = _node(graph, "pkg.api", EvidenceNodeKind.FILE)
    build_function = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)
    core_file = _node(graph, "pkg.core", EvidenceNodeKind.FILE)
    json_module = _node(graph, "json", EvidenceNodeKind.EXTERNAL_MODULE)

    import_targets = {
        graph.node_by_entity_id(edge.target_entity_id).entity_id
        for edge in graph.edges
        if edge.relation is EvidenceRelationKind.IMPORTS
        and edge.source_entity_id == api_file.entity_id
    }
    assert core_file.entity_id in import_targets
    assert json_module.entity_id in import_targets

    service_definition = _node(graph, "pkg.core.Service", EvidenceNodeKind.CLASS)
    service_reference = _node(graph, "Service", EvidenceNodeKind.UNRESOLVED_SYMBOL)
    reference_targets = {
        graph.node_by_entity_id(edge.target_entity_id).entity_id
        for edge in graph.edges
        if edge.relation is EvidenceRelationKind.REFERENCES
        and edge.source_entity_id == build_function.entity_id
    }
    assert service_reference.entity_id in reference_targets
    assert service_definition.entity_id not in reference_targets


def test_builder_unions_duplicate_reference_evidence_without_duplicate_assertions(
    tmp_path: Path,
) -> None:
    graph = build_evidence_graph(_indexes(tmp_path))
    run_node = _node(graph, "pkg.core.Service.run", EvidenceNodeKind.METHOD)
    helper_node = _node(graph, "helper", EvidenceNodeKind.UNRESOLVED_SYMBOL)
    edges = [
        edge
        for edge in graph.edges
        if edge.relation is EvidenceRelationKind.REFERENCES
        and edge.source_entity_id == run_node.entity_id
        and edge.target_entity_id == helper_node.entity_id
    ]

    assert len(edges) == 1
    assert len(edges[0].evidence) >= 2
    assert len({item.evidence_id for item in edges[0].evidence}) == len(
        edges[0].evidence
    )


def test_builder_rejects_empty_duplicate_or_mixed_snapshot_indexes(
    tmp_path: Path,
) -> None:
    indexes = _indexes(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        build_evidence_graph(())
    with pytest.raises(ValueError, match="duplicate.*path"):
        build_evidence_graph((indexes[0], indexes[0]))

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_source = other_root / "other.py"
    other_source.write_text("value = 1\n", encoding="utf-8")
    other = build_python_file_index(
        other_root,
        Path("other.py"),
        repository_id=hashlib.sha256(b"other-repository").hexdigest(),
        commit_sha=COMMIT_SHA,
    )
    with pytest.raises(ValueError, match="snapshot"):
        build_evidence_graph((indexes[0], other))
