"""Tests for bounded graph slices, reading steps, and claim lineage."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.structure import (
    EvidenceGraph,
    EvidenceGraphSlice,
    EvidenceNodeKind,
    EvidenceRelationKind,
    GraphSliceTruncationReason,
    GraphTraversalDirection,
    ReadingPathError,
    ReadingPathErrorReason,
    ReadingStepKind,
    StructuralVerificationStatus,
    TraceableReadingPath,
    build_evidence_graph,
    build_python_file_index,
    build_traceable_reading_path,
    create_evidence_graph,
    create_graph_edge,
    create_graph_node,
    create_traceable_claim,
    resolve_seed_entity_ids,
    slice_evidence_graph,
)


REPOSITORY_ID = hashlib.sha256(b"reading-path-repository").hexdigest()
COMMIT_SHA = "e" * 40


def _graph(tmp_path: Path) -> EvidenceGraph:
    sources = {
        "pkg/core.py": (
            "def helper():\n"
            "    return True\n\n"
            "class Service:\n"
            "    def run(self):\n"
            "        return helper()\n"
        ),
        "pkg/api.py": (
            "from .core import Service\n\ndef build():\n    return Service().run()\n"
        ),
        "tests/test_api.py": (
            "from pkg.api import build\n\ndef test_build():\n    assert build()\n"
        ),
    }
    indexes = []
    for relative_path, source in sources.items():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        indexes.append(
            build_python_file_index(
                tmp_path,
                Path(relative_path),
                repository_id=REPOSITORY_ID,
                commit_sha=COMMIT_SHA,
            )
        )
    return build_evidence_graph(tuple(indexes))


def _node(graph: EvidenceGraph, name: str, kind: EvidenceNodeKind):
    return next(
        node
        for node in graph.nodes
        if node.canonical_name == name and node.kind is kind
    )


def _import_edge(graph: EvidenceGraph):
    source = _node(graph, "pkg.api", EvidenceNodeKind.FILE)
    target = _node(graph, "pkg.core", EvidenceNodeKind.FILE)
    return next(
        edge
        for edge in graph.edges
        if edge.relation is EvidenceRelationKind.IMPORTS
        and edge.source_entity_id == source.entity_id
        and edge.target_entity_id == target.entity_id
    )


def _augmented_graph(graph: EvidenceGraph) -> tuple[EvidenceGraph, object, object]:
    build = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)
    api_file = _node(graph, "pkg.api", EvidenceNodeKind.FILE)
    candidate = create_graph_node(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        kind=EvidenceNodeKind.UNRESOLVED_SYMBOL,
        canonical_name="candidate_service",
    )
    inferred = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=EvidenceRelationKind.REFERENCES,
        source_entity_id=build.entity_id,
        target_entity_id=candidate.entity_id,
        verification_status=StructuralVerificationStatus.INFERRED,
        explanation="candidate inferred from task wording",
    )
    rejected = create_graph_edge(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        relation=EvidenceRelationKind.REFERENCES,
        source_entity_id=api_file.entity_id,
        target_entity_id=candidate.entity_id,
        verification_status=StructuralVerificationStatus.REJECTED,
        explanation="AST scope contains no matching reference",
    )
    return (
        create_evidence_graph(
            repository_id=graph.repository_id,
            commit_sha=graph.commit_sha,
            nodes=(*graph.nodes, candidate),
            edges=(*graph.edges, inferred, rejected),
        ),
        inferred,
        rejected,
    )


def test_reading_path_is_deterministic_strict_frozen_and_fully_traceable(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    build = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)
    import_edge = _import_edge(graph)
    claim = create_traceable_claim(
        text="The API module imports the core implementation.",
        supporting_assertion_ids=(import_edge.assertion_id,),
    )

    path = build_traceable_reading_path(
        graph,
        objective="Trace how the public builder reaches the core service.",
        seed_entity_ids=(build.entity_id,),
        claims=(claim,),
        max_depth=3,
        max_nodes=64,
        max_edges=128,
    )

    assert path == build_traceable_reading_path(
        graph,
        objective=path.objective,
        seed_entity_ids=(build.entity_id,),
        claims=(claim,),
        max_depth=3,
        max_nodes=64,
        max_edges=128,
    )
    assert TraceableReadingPath.model_validate_json(path.model_dump_json()) == path
    assert path.graph_slice.graph_id == graph.graph_id
    assert (
        path.claim_lineages[0].verification_status
        is StructuralVerificationStatus.VERIFIED
    )
    assert path.claim_lineages[0].provenance_complete is True
    assert path.claim_lineages[0].evidence
    assert path.claim_lineages[0].source_locations
    assert all(step.source_locations for step in path.steps)
    assert all(
        location.path == step.path
        for step in path.steps
        for location in step.source_locations
    )
    assert [step.ordinal for step in path.steps] == list(range(1, len(path.steps) + 1))
    with pytest.raises(ValidationError, match="frozen"):
        path.objective = "changed"  # type: ignore[misc]


def test_rejected_edges_never_enter_slice_and_inference_stays_explicit(
    tmp_path: Path,
) -> None:
    graph, inferred, rejected = _augmented_graph(_graph(tmp_path))
    build = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)
    claim = create_traceable_claim(
        text="The builder may reference the candidate service.",
        supporting_assertion_ids=(inferred.assertion_id,),
        explanation="Only task wording supports this candidate relation.",
    )

    default_slice = slice_evidence_graph(
        graph,
        seed_entity_ids=(build.entity_id,),
        max_depth=1,
    )
    assert inferred.assertion_id not in {
        edge.assertion_id for edge in default_slice.edges
    }

    path = build_traceable_reading_path(
        graph,
        objective="Inspect a tentative candidate.",
        seed_entity_ids=(build.entity_id,),
        claims=(claim,),
        max_depth=1,
        max_nodes=32,
        max_edges=64,
        include_inferred=True,
    )

    assertion_ids = {edge.assertion_id for edge in path.graph_slice.edges}
    assert inferred.assertion_id in assertion_ids
    assert rejected.assertion_id not in assertion_ids
    lineage = path.claim_lineages[0]
    assert lineage.verification_status is StructuralVerificationStatus.INFERRED
    assert lineage.provenance_complete is False
    assert lineage.evidence == ()
    assert lineage.source_locations == ()
    assert lineage.explanation == claim.explanation

    with pytest.raises(ReadingPathError) as excluded:
        build_traceable_reading_path(
            graph,
            objective="Exclude inference.",
            seed_entity_ids=(build.entity_id,),
            claims=(claim,),
            max_depth=1,
            include_inferred=False,
        )
    assert excluded.value.reason is ReadingPathErrorReason.ASSERTION_OUTSIDE_SLICE

    unexplained = create_traceable_claim(
        text="Unexplained inferred claim.",
        supporting_assertion_ids=(inferred.assertion_id,),
    )
    with pytest.raises(ReadingPathError) as missing_explanation:
        build_traceable_reading_path(
            graph,
            objective="Reject unexplained inference.",
            seed_entity_ids=(build.entity_id,),
            claims=(unexplained,),
            max_depth=1,
            include_inferred=True,
        )
    assert (
        missing_explanation.value.reason
        is ReadingPathErrorReason.INFERRED_CLAIM_EXPLANATION_REQUIRED
    )


def test_claim_requires_at_least_one_canonical_assertion() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        create_traceable_claim(text="Unsupported claim.", supporting_assertion_ids=())

    assertion_a = "a" * 64
    assertion_b = "b" * 64
    claim = create_traceable_claim(
        text="Canonical claim.",
        supporting_assertion_ids=(assertion_b, assertion_a),
    )
    assert claim.supporting_assertion_ids == (assertion_a, assertion_b)


def test_seed_resolution_rejects_unknown_ambiguous_and_duplicate_selectors(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    unresolved = _node(graph, "Service", EvidenceNodeKind.UNRESOLVED_SYMBOL)
    external = create_graph_node(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        kind=EvidenceNodeKind.EXTERNAL_MODULE,
        canonical_name="Service",
    )
    ambiguous_graph = create_evidence_graph(
        repository_id=graph.repository_id,
        commit_sha=graph.commit_sha,
        nodes=(*graph.nodes, external),
        edges=graph.edges,
    )

    with pytest.raises(ReadingPathError) as ambiguous:
        resolve_seed_entity_ids(ambiguous_graph, ("Service",))
    assert ambiguous.value.reason is ReadingPathErrorReason.AMBIGUOUS_SEED
    with pytest.raises(ReadingPathError) as unknown:
        resolve_seed_entity_ids(graph, ("missing.symbol",))
    assert unknown.value.reason is ReadingPathErrorReason.UNKNOWN_SEED
    with pytest.raises(ReadingPathError) as duplicate:
        resolve_seed_entity_ids(graph, (unresolved.entity_id, "Service"))
    assert duplicate.value.reason is ReadingPathErrorReason.DUPLICATE_SEED
    assert resolve_seed_entity_ids(graph, (unresolved.entity_id,)) == (
        unresolved.entity_id,
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"max_depth": 0}, GraphSliceTruncationReason.DEPTH_LIMIT),
        ({"max_nodes": 1}, GraphSliceTruncationReason.NODE_LIMIT),
        ({"max_edges": 0}, GraphSliceTruncationReason.EDGE_LIMIT),
    ],
)
def test_slice_terminates_on_cycles_and_reports_each_bound(
    tmp_path: Path,
    kwargs: dict[str, int],
    reason: GraphSliceTruncationReason,
) -> None:
    graph = _graph(tmp_path)
    build = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)

    graph_slice = slice_evidence_graph(
        graph,
        seed_entity_ids=(build.entity_id,),
        traversal=GraphTraversalDirection.BOTH,
        max_depth=kwargs.get("max_depth", 8),
        max_nodes=kwargs.get("max_nodes", 64),
        max_edges=kwargs.get("max_edges", 128),
    )

    assert graph_slice.truncated is True
    assert reason in graph_slice.truncation_reasons
    assert len({node.entity_id for node in graph_slice.nodes}) == len(graph_slice.nodes)
    assert len({edge.assertion_id for edge in graph_slice.edges}) == len(
        graph_slice.edges
    )


def test_reading_order_places_files_before_symbols_and_tests_last(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    path = build_traceable_reading_path(
        graph,
        objective="Read the repository slice.",
        seed_entity_ids=(graph.repository_node.entity_id,),
        max_depth=3,
        max_nodes=128,
        max_edges=256,
    )
    ranks = {
        ReadingStepKind.FILE: 0,
        ReadingStepKind.SYMBOL: 1,
        ReadingStepKind.TEST: 2,
    }

    assert [ranks[step.kind] for step in path.steps] == sorted(
        ranks[step.kind] for step in path.steps
    )
    assert any(step.kind is ReadingStepKind.FILE for step in path.steps)
    assert any(step.kind is ReadingStepKind.SYMBOL for step in path.steps)
    assert path.steps[-1].kind is ReadingStepKind.TEST


@pytest.mark.parametrize(
    ("traversal", "seed_name", "seed_kind"),
    [
        (
            GraphTraversalDirection.OUTBOUND,
            "pkg.api",
            EvidenceNodeKind.FILE,
        ),
        (
            GraphTraversalDirection.INBOUND,
            "pkg.api.build",
            EvidenceNodeKind.FUNCTION,
        ),
    ],
)
def test_directional_slice_preserves_depth_decreasing_step_lineage(
    tmp_path: Path,
    traversal: GraphTraversalDirection,
    seed_name: str,
    seed_kind: EvidenceNodeKind,
) -> None:
    graph = _graph(tmp_path)
    seed = _node(graph, seed_name, seed_kind)

    path = build_traceable_reading_path(
        graph,
        objective="Verify directional predecessor lineage.",
        seed_entity_ids=(seed.entity_id,),
        traversal=traversal,
        max_depth=1,
    )

    assert all(step.depth == 0 or step.via_assertion_id for step in path.steps)


def test_slice_and_reading_path_reject_tampering_and_noncanonical_order(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    build = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)
    path = build_traceable_reading_path(
        graph,
        objective="Tamper test.",
        seed_entity_ids=(build.entity_id,),
        max_depth=2,
    )

    payload = path.model_dump(mode="json")
    with pytest.raises(ValidationError, match="reading_path_id"):
        TraceableReadingPath.model_validate_json(
            json.dumps({**payload, "reading_path_id": "f" * 64})
        )

    reversed_steps = list(reversed(payload["steps"]))
    with pytest.raises(ValidationError, match="ordinal|canonical"):
        TraceableReadingPath.model_validate_json(
            json.dumps({**payload, "steps": reversed_steps})
        )

    slice_payload = path.graph_slice.model_dump(mode="json")
    with pytest.raises(ValidationError, match="slice_id"):
        EvidenceGraphSlice.model_validate_json(
            json.dumps({**slice_payload, "slice_id": "f" * 64})
        )
    with pytest.raises(ValidationError, match="extra"):
        EvidenceGraphSlice.model_validate_json(
            json.dumps({**slice_payload, "api_key": "secret"})
        )
    with pytest.raises(ValidationError):
        EvidenceGraphSlice.model_validate_json(
            json.dumps({**slice_payload, "max_depth": str(slice_payload["max_depth"])})
        )
    with pytest.raises(ValidationError):
        EvidenceGraphSlice.model_validate_json(
            json.dumps({**slice_payload, "schema_version": 2})
        )
    with pytest.raises(ValidationError, match="canonical"):
        EvidenceGraphSlice.model_validate_json(
            json.dumps(
                {
                    **slice_payload,
                    "nodes": list(reversed(slice_payload["nodes"])),
                }
            )
        )

    steps = list(payload["steps"])
    step = dict(steps[0])
    locations = list(step["source_locations"])
    locations[0] = {**locations[0], "line_end": 999}
    step["source_locations"] = locations
    steps[0] = step
    with pytest.raises(ValidationError, match="location_id|source"):
        TraceableReadingPath.model_validate_json(
            json.dumps({**payload, "steps": steps})
        )


def test_slice_rejects_unknown_seed_and_seed_count_over_node_limit(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    build = _node(graph, "pkg.api.build", EvidenceNodeKind.FUNCTION)
    core = _node(graph, "pkg.core", EvidenceNodeKind.FILE)

    with pytest.raises(ReadingPathError) as unknown:
        slice_evidence_graph(graph, seed_entity_ids=("f" * 64,))
    assert unknown.value.reason is ReadingPathErrorReason.UNKNOWN_SEED
    with pytest.raises(ReadingPathError) as limit:
        slice_evidence_graph(
            graph,
            seed_entity_ids=(build.entity_id, core.entity_id),
            max_nodes=1,
        )
    assert limit.value.reason is ReadingPathErrorReason.SEED_LIMIT
