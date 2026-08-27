"""Generate or verify the deterministic M4.3 UI/MCP reading-path artifact."""

import argparse
import hashlib
import json
from pathlib import Path

from repocompass.structure import (
    EvidenceNodeKind,
    EvidenceRelationKind,
    build_evidence_graph,
    build_python_file_index,
    build_traceable_reading_path,
    create_traceable_claim,
)

REPOSITORY_ID = hashlib.sha256(b"repocompass-reading-path-fixture").hexdigest()
COMMIT_SHA = "1" * 40
FIXTURE_DIRECTORY = Path(__file__).with_name("reading_path_project")
ARTIFACT_PATH = Path(__file__).with_name("reading_path_artifact.json")


def render_artifact() -> str:
    """Build the public artifact without timestamps or machine-local paths."""

    indexes = tuple(
        build_python_file_index(
            FIXTURE_DIRECTORY,
            source.relative_to(FIXTURE_DIRECTORY),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
        for source in sorted(FIXTURE_DIRECTORY.rglob("*.py"))
    )
    graph = build_evidence_graph(indexes)
    build_node = next(
        node
        for node in graph.nodes
        if node.kind is EvidenceNodeKind.FUNCTION
        and node.canonical_name == "pkg.api.build"
    )
    api_file = next(
        node
        for node in graph.nodes
        if node.kind is EvidenceNodeKind.FILE and node.canonical_name == "pkg.api"
    )
    core_file = next(
        node
        for node in graph.nodes
        if node.kind is EvidenceNodeKind.FILE and node.canonical_name == "pkg.core"
    )
    import_edge = next(
        edge
        for edge in graph.edges
        if edge.relation is EvidenceRelationKind.IMPORTS
        and edge.source_entity_id == api_file.entity_id
        and edge.target_entity_id == core_file.entity_id
    )
    claim = create_traceable_claim(
        text="The API module imports the core implementation.",
        supporting_assertion_ids=(import_edge.assertion_id,),
    )
    reading_path = build_traceable_reading_path(
        graph,
        objective="Trace how the public builder reaches the core service.",
        seed_entity_ids=(build_node.entity_id,),
        claims=(claim,),
        max_depth=3,
        max_nodes=64,
        max_edges=128,
    )
    return (
        json.dumps(
            reading_path.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed artifact is not deterministic and current",
    )
    args = parser.parse_args()
    rendered = render_artifact()
    if args.check:
        if (
            not ARTIFACT_PATH.is_file()
            or ARTIFACT_PATH.read_text(encoding="utf-8") != rendered
        ):
            parser.error("reading_path_artifact.json is missing or stale")
        return 0
    ARTIFACT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
