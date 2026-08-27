"""Deterministic conversion from bounded Python AST facts to Evidence Graph."""

from pathlib import PurePosixPath

from repocompass.structure.contracts import (
    PythonFileIndex,
    PythonImportKind,
    PythonImportRecord,
    PythonSymbolKind,
    PythonSymbolRecord,
    StructuralVerificationStatus,
)
from repocompass.structure.evidence_graph import (
    MAX_EVIDENCE_GRAPH_EDGES,
    MAX_EVIDENCE_GRAPH_NODES,
    EvidenceGraph,
    EvidenceGraphNode,
    EvidenceNodeKind,
    EvidenceRelationKind,
    GraphEvidence,
    create_evidence_graph,
    create_graph_edge,
    create_graph_evidence,
    create_graph_node,
)


def build_evidence_graph(indexes: tuple[PythonFileIndex, ...]) -> EvidenceGraph:
    """Build a conservative graph for one immutable Python repository snapshot."""

    if not isinstance(indexes, tuple):
        raise TypeError("indexes must be a tuple of PythonFileIndex instances")
    if not indexes:
        raise ValueError("at least one Python file index is required")
    if any(not isinstance(index, PythonFileIndex) for index in indexes):
        raise TypeError("indexes must contain only PythonFileIndex instances")

    repository_id = indexes[0].repository_id
    commit_sha = indexes[0].commit_sha
    if any(
        index.repository_id != repository_id or index.commit_sha != commit_sha
        for index in indexes[1:]
    ):
        raise ValueError("all indexes must belong to the same repository snapshot")
    paths = tuple(index.path for index in indexes)
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate Python index path")

    ordered_indexes = tuple(sorted(indexes, key=lambda index: index.path))
    repository_node = create_graph_node(
        repository_id=repository_id,
        commit_sha=commit_sha,
        kind=EvidenceNodeKind.REPOSITORY,
        canonical_name="repository",
    )
    nodes_by_entity: dict[str, EvidenceGraphNode] = {
        repository_node.entity_id: repository_node
    }
    nodes_by_symbol: dict[str, EvidenceGraphNode] = {}
    files_by_module: dict[str, EvidenceGraphNode] = {}
    index_by_module: dict[str, PythonFileIndex] = {}

    for index in ordered_indexes:
        if index.module_name in index_by_module:
            raise ValueError("duplicate Python module name is ambiguous")
        index_by_module[index.module_name] = index
        for symbol in index.symbols:
            if symbol.symbol_id in nodes_by_symbol:
                raise ValueError("duplicate Python symbol identity")
            evidence = create_graph_evidence(symbol)
            node = create_graph_node(
                repository_id=repository_id,
                commit_sha=commit_sha,
                kind=_node_kind(symbol),
                canonical_name=symbol.qualified_name,
                path=index.path,
                symbol_id=symbol.symbol_id,
                symbol_kind=symbol.kind,
                is_test=symbol.is_test and symbol.kind is not PythonSymbolKind.MODULE,
                evidence=(evidence,),
            )
            _add_node(nodes_by_entity, node)
            nodes_by_symbol[symbol.symbol_id] = node
            if symbol.kind is PythonSymbolKind.MODULE:
                files_by_module[index.module_name] = node

    assertions: dict[
        tuple[EvidenceRelationKind, str, str],
        dict[str, GraphEvidence],
    ] = {}
    for index in ordered_indexes:
        for symbol in index.symbols:
            target = nodes_by_symbol[symbol.symbol_id]
            source = (
                repository_node
                if symbol.kind is PythonSymbolKind.MODULE
                else nodes_by_symbol[symbol.parent_symbol_id or ""]
            )
            _add_assertion_evidence(
                assertions,
                relation=EvidenceRelationKind.CONTAINS,
                source=source,
                target=target,
                evidence=create_graph_evidence(symbol),
            )

        for record in index.imports:
            source = nodes_by_symbol[record.scope_symbol_id]
            target = _import_target_node(
                index=index,
                record=record,
                files_by_module=files_by_module,
                repository_id=repository_id,
                commit_sha=commit_sha,
            )
            _add_node(nodes_by_entity, target)
            _add_assertion_evidence(
                assertions,
                relation=EvidenceRelationKind.IMPORTS,
                source=source,
                target=target,
                evidence=create_graph_evidence(record),
            )

        for record in index.references:
            source = nodes_by_symbol[record.scope_symbol_id]
            target = create_graph_node(
                repository_id=repository_id,
                commit_sha=commit_sha,
                kind=EvidenceNodeKind.UNRESOLVED_SYMBOL,
                canonical_name=record.name,
            )
            _add_node(nodes_by_entity, target)
            _add_assertion_evidence(
                assertions,
                relation=EvidenceRelationKind.REFERENCES,
                source=source,
                target=target,
                evidence=create_graph_evidence(record),
            )

    if len(assertions) > MAX_EVIDENCE_GRAPH_EDGES:
        raise ValueError("evidence graph exceeds the edge limit")
    edges = tuple(
        create_graph_edge(
            repository_id=repository_id,
            commit_sha=commit_sha,
            relation=relation,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            verification_status=StructuralVerificationStatus.VERIFIED,
            evidence=tuple(evidence_by_id.values()),
        )
        for (
            relation,
            source_entity_id,
            target_entity_id,
        ), evidence_by_id in assertions.items()
    )
    return create_evidence_graph(
        repository_id=repository_id,
        commit_sha=commit_sha,
        nodes=tuple(nodes_by_entity.values()),
        edges=edges,
    )


def _node_kind(symbol: PythonSymbolRecord) -> EvidenceNodeKind:
    if symbol.kind is PythonSymbolKind.MODULE:
        return EvidenceNodeKind.FILE
    if symbol.is_test:
        return EvidenceNodeKind.TEST
    if symbol.kind is PythonSymbolKind.CLASS:
        return EvidenceNodeKind.CLASS
    if symbol.kind in {
        PythonSymbolKind.FUNCTION,
        PythonSymbolKind.ASYNC_FUNCTION,
    }:
        return EvidenceNodeKind.FUNCTION
    return EvidenceNodeKind.METHOD


def _import_target_node(
    *,
    index: PythonFileIndex,
    record: PythonImportRecord,
    files_by_module: dict[str, EvidenceGraphNode],
    repository_id: str,
    commit_sha: str | None,
) -> EvidenceGraphNode:
    resolved_module = _resolve_import_module(index, record)
    candidates = [resolved_module] if resolved_module else []
    if (
        record.kind is PythonImportKind.FROM_IMPORT
        and record.imported_name != "*"
        and resolved_module
    ):
        candidates.append(f"{resolved_module}.{record.imported_name}")
    for candidate in candidates:
        internal = files_by_module.get(candidate)
        if internal is not None:
            return internal

    display_name = resolved_module or (
        "." * record.level + (record.module or record.imported_name)
    )
    return create_graph_node(
        repository_id=repository_id,
        commit_sha=commit_sha,
        kind=EvidenceNodeKind.EXTERNAL_MODULE,
        canonical_name=display_name,
    )


def _resolve_import_module(
    index: PythonFileIndex,
    record: PythonImportRecord,
) -> str | None:
    if record.level == 0:
        return record.module
    module_parts = index.module_name.split(".")
    is_package_init = PurePosixPath(index.path).name == "__init__.py"
    package_parts = module_parts if is_package_init else module_parts[:-1]
    parent_hops = record.level - 1
    if parent_hops > len(package_parts):
        return None
    if parent_hops:
        package_parts = package_parts[:-parent_hops]
    imported_module_parts = [] if record.module is None else record.module.split(".")
    resolved_parts = [*package_parts, *imported_module_parts]
    return ".".join(part for part in resolved_parts if part) or None


def _add_node(
    nodes_by_entity: dict[str, EvidenceGraphNode],
    node: EvidenceGraphNode,
) -> None:
    previous = nodes_by_entity.get(node.entity_id)
    if previous is not None and previous != node:
        raise ValueError("duplicate graph entity has conflicting contents")
    nodes_by_entity[node.entity_id] = node
    if len(nodes_by_entity) > MAX_EVIDENCE_GRAPH_NODES:
        raise ValueError("evidence graph exceeds the node limit")


def _add_assertion_evidence(
    assertions: dict[
        tuple[EvidenceRelationKind, str, str],
        dict[str, GraphEvidence],
    ],
    *,
    relation: EvidenceRelationKind,
    source: EvidenceGraphNode,
    target: EvidenceGraphNode,
    evidence: GraphEvidence,
) -> None:
    key = (relation, source.entity_id, target.entity_id)
    evidence_by_id = assertions.setdefault(key, {})
    previous = evidence_by_id.get(evidence.evidence_id)
    if previous is not None and previous != evidence:
        raise ValueError("duplicate evidence identity has conflicting contents")
    evidence_by_id[evidence.evidence_id] = evidence
