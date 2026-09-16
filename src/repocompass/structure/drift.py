"""Bounded, read-only provenance drift; unchanged bytes are not semantic proof."""

import hashlib
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from repocompass.memory.revalidation import (
    MAX_MEMORY_SOURCE_BYTES,
    MemorySourceReadError,
    read_memory_source_bytes,
    repository_snapshot_is_clean,
)
from repocompass.tools.contracts import CommitSha, Sha256Digest
from repocompass.tools.gateway import ToolExecutionContext

MAX_DRIFT_FILES = 64
MAX_DRIFT_BYTES = 16 * 1024 * 1024


class _DriftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceAnchor(_DriftModel):
    path: str = Field(min_length=1, max_length=4096)
    digest: Sha256Digest


class DriftSubject(_DriftModel):
    id: str = Field(min_length=1, max_length=256)
    kind: Literal["claim", "entity", "memory"]
    paths: tuple[str, ...] = Field(max_length=MAX_DRIFT_FILES)
    provenance_complete: bool = True


class DriftBaseline(_DriftModel):
    repository_id: Sha256Digest
    commit_sha: CommitSha | None
    anchors: tuple[SourceAnchor, ...] = Field(max_length=1024)
    subjects: tuple[DriftSubject, ...] = Field(max_length=2048)
    # (dependency, dependent) path pairs, derived only from persisted graph edges.
    dependencies: tuple[tuple[str, str], ...] = Field(default=(), max_length=10000)
    scope_truncated: bool = False


class SourceDrift(_DriftModel):
    path: str
    baseline_digests: tuple[Sha256Digest, ...]
    current_digest: Sha256Digest | None
    status: Literal[
        "unchanged", "changed", "unavailable", "conflicting_baseline", "not_checked"
    ]
    reason: str | None = None
    bytes_read: int = Field(ge=0)


class SubjectDrift(_DriftModel):
    id: str
    kind: Literal["claim", "entity", "memory"]
    status: Literal[
        "inputs_unchanged",
        "revalidation_required",
        "evidence_invalidated",
        "unverifiable",
    ]
    paths: tuple[str, ...]
    reasons: tuple[str, ...]


class EvidenceDriftReport(_DriftModel):
    schema_version: Literal[1] = 1
    repository_id: Sha256Digest
    baseline_commit: CommitSha | None
    current_commit: CommitSha | None
    worktree_clean: bool
    snapshot_stable: bool
    changes_outside_scope: bool | None
    scope_truncated: bool
    sources: tuple[SourceDrift, ...]
    subjects: tuple[SubjectDrift, ...]
    recheck_paths: tuple[str, ...]
    full_recheck_paths: tuple[str, ...]
    bytes_read: int
    read_budget_used_bytes: int
    file_reads: int
    provider_calls: Literal[0] = 0
    disclaimer: Literal["provenance_only_not_semantic_revalidation"] = (
        "provenance_only_not_semantic_revalidation"
    )


def analyze_evidence_drift(
    baseline: DriftBaseline, repository_root: Path
) -> EvidenceDriftReport:
    before = ToolExecutionContext(repository_root)
    if before.repository_id != baseline.repository_id:
        raise ValueError("drift baseline belongs to another repository")
    clean_before = repository_snapshot_is_clean(before)
    expected: dict[str, set[str]] = {}
    for anchor in baseline.anchors:
        expected.setdefault(anchor.path, set()).add(anchor.digest)
    paths = sorted(expected)
    truncated = baseline.scope_truncated or len(paths) > MAX_DRIFT_FILES
    records = []
    remaining = MAX_DRIFT_BYTES
    reads = 0
    for index, path in enumerate(paths):
        digests = tuple(sorted(expected[path]))
        reason = digest = None
        size = 0
        if len(digests) != 1:
            status, reason = "conflicting_baseline", "multiple_digests_for_one_path"
        elif index >= MAX_DRIFT_FILES or remaining <= 1:
            status, reason = "not_checked", "read_budget"
            truncated = True
        else:
            limit = min(remaining - 1, MAX_MEMORY_SOURCE_BYTES)
            # Reserve even failed/racing reads, including the overflow sentinel byte.
            remaining -= limit + 1
            try:
                reads += 1
                content = read_memory_source_bytes(
                    repository_root, path, max_bytes=limit
                )
                size = len(content)
                remaining += limit + 1 - size
                digest = hashlib.sha256(content).hexdigest()
                status = "unchanged" if digest == digests[0] else "changed"
            except MemorySourceReadError as error:
                status, reason = "unavailable", error.reason.value
        records.append(
            SourceDrift(
                path=path,
                baseline_digests=digests,
                current_digest=digest,
                status=status,
                reason=reason,
                bytes_read=size,
            )
        )
    after = ToolExecutionContext(repository_root)
    clean_after = repository_snapshot_is_clean(after)
    stable = (
        before.commit_sha == after.commit_sha
        and before.repository_id == after.repository_id
        and clean_before
        and clean_after
    )
    changed = {record.path for record in records if record.status != "unchanged"}
    # Reverse closure over recorded import/reference dependencies only; not whole-repo reachability.
    affected = set(changed)
    while True:
        expanded = affected | {
            dependent
            for dependency, dependent in baseline.dependencies
            if dependency in affected
        }
        if expanded == affected:
            break
        affected = expanded
    by_path = {record.path: record for record in records}
    subjects = []
    for subject in baseline.subjects:
        reasons = []
        if (
            not subject.provenance_complete
            or not subject.paths
            or any(path not in by_path for path in subject.paths)
        ):
            status = "unverifiable"
            reasons.append("incomplete_provenance")
        elif any(
            by_path[path].status
            in ("unavailable", "conflicting_baseline", "not_checked")
            for path in subject.paths
        ):
            status = "unverifiable"
            reasons.append("source_unavailable")
        elif any(path in changed for path in subject.paths):
            status = "evidence_invalidated"
            reasons.append("source_digest_changed")
        elif any(path in affected for path in subject.paths):
            status = "revalidation_required"
            reasons.append("recorded_dependency_changed")
        elif (
            baseline.commit_sha is None
            or baseline.commit_sha != after.commit_sha
            or not stable
        ):
            status = "revalidation_required"
            reasons.append("snapshot_changed_or_unstable")
        else:
            status = "inputs_unchanged"
            reasons.append("recorded_source_digests_match")
        subjects.append(
            SubjectDrift(
                id=subject.id,
                kind=subject.kind,
                status=status,
                paths=subject.paths,
                reasons=tuple(reasons),
            )
        )
    # Dirty/racing snapshots or incomplete/truncated coverage cannot justify a narrow recommendation.
    outside = _changes_outside_scope(
        repository_root, baseline.commit_sha, after.commit_sha, paths
    )
    uncertain = (
        truncated
        or not stable
        or outside is not False
        or any(subject.status == "unverifiable" for subject in subjects)
    )
    recheck = set(paths) if uncertain else affected.intersection(paths)
    return EvidenceDriftReport(
        repository_id=before.repository_id,
        baseline_commit=baseline.commit_sha,
        current_commit=after.commit_sha,
        worktree_clean=clean_after,
        snapshot_stable=stable,
        changes_outside_scope=outside,
        scope_truncated=truncated,
        sources=tuple(records),
        subjects=tuple(subjects),
        recheck_paths=tuple(sorted(recheck)),
        full_recheck_paths=tuple(paths),
        bytes_read=sum(record.bytes_read for record in records),
        read_budget_used_bytes=MAX_DRIFT_BYTES - remaining,
        file_reads=reads,
    )


def baseline_from_structure(
    structure, memories=(), *, scope_truncated=False
) -> DriftBaseline:
    """Project persisted anchors, never model text or arbitrary browser paths."""
    graph = structure.reading_path.graph_slice
    anchors = {
        (evidence.location.path, evidence.location.source_digest)
        for node in graph.nodes
        for evidence in node.evidence
    }
    anchors.update(
        (evidence.location.path, evidence.location.source_digest)
        for edge in graph.edges
        for evidence in edge.evidence
    )
    anchors.update(
        (location.path, location.source_digest)
        for step in structure.reading_path.steps
        for location in step.source_locations
    )
    known_paths = {path for path, _ in anchors}
    subjects = [
        DriftSubject(id=symbol.entity_id, kind="entity", paths=(symbol.path,))
        for symbol in structure.reading_path.steps
    ]
    subjects.extend(
        DriftSubject(
            id=claim.claim_id,
            kind="claim",
            paths=tuple(sorted({source.path for source in claim.source_locations})),
            provenance_complete=claim.provenance_complete,
        )
        for claim in structure.reading_path.claim_lineages
    )
    for memory in memories:
        paths = {source.location.path for source in memory.sources}
        # Only same-snapshot facts wholly grounded in this already-public task slice.
        if (
            memory.source_repository_id != structure.repository_id
            or memory.commit_sha != structure.commit_sha
            or not paths <= known_paths
        ):
            continue
        subjects.append(
            DriftSubject(id=memory.memory_id, kind="memory", paths=tuple(sorted(paths)))
        )
        anchors.update(
            (source.location.path, source.source_digest) for source in memory.sources
        )
    nodes = {node.entity_id: node for node in graph.nodes}
    dependencies = set()
    for edge in graph.edges:
        if edge.relation.value not in ("imports", "references"):
            continue
        source, target = nodes[edge.source_entity_id], nodes[edge.target_entity_id]
        if (
            source.path in known_paths
            and target.path in known_paths
            and source.path != target.path
        ):
            dependencies.add((target.path, source.path))
    ordered_anchors = sorted(anchors)
    return DriftBaseline(
        repository_id=structure.repository_id,
        commit_sha=structure.commit_sha,
        anchors=tuple(
            SourceAnchor(path=path, digest=digest)
            for path, digest in ordered_anchors[:1024]
        ),
        subjects=tuple(subjects[:2048]),
        dependencies=tuple(sorted(dependencies)[:10000]),
        scope_truncated=scope_truncated
        or len(anchors) > 1024
        or len(subjects) > 2048
        or len(dependencies) > 10000
        or structure.reading_path.graph_slice.truncated,
    )


def _changes_outside_scope(
    root: Path, baseline: str | None, current: str | None, paths: list[str]
) -> bool | None:
    if baseline is None or current is None:
        return None
    if baseline == current:
        return False
    if len(paths) > MAX_DRIFT_FILES:
        return None
    # Constant command, validated commit hashes, literal excludes. No source output,
    # external diff driver, textconv, shell, checkout, or network operation.
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--quiet",
                baseline,
                current,
                "--",
                ".",
                *[":(exclude,literal)" + path for path in paths],
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {0: False, 1: True}.get(result.returncode)
