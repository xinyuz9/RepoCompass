"""Independent hand-labelled source/dependency cases for drift decisions."""

import hashlib
import subprocess

import pytest

from repocompass.structure.drift import (
    DriftBaseline,
    DriftSubject,
    SourceAnchor,
    analyze_evidence_drift,
)
from repocompass.tools.gateway import ToolExecutionContext


def commit(repo):
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


@pytest.fixture
def drift_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    anchors = []
    for name in ("a.py", "b.py", "c.py"):
        (repo / name).write_bytes(b"VALUE = 1\n")
        anchors.append(
            SourceAnchor(path=name, digest=hashlib.sha256(b"VALUE = 1\n").hexdigest())
        )
    commit(repo)
    context = ToolExecutionContext(repo)
    return repo, DriftBaseline(
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
        anchors=tuple(anchors),
        dependencies=(("a.py", "b.py"),),
        subjects=(
            DriftSubject(id="claim-a", kind="claim", paths=("a.py",)),
            DriftSubject(id="entity-b", kind="entity", paths=("b.py",)),
            DriftSubject(id="memory-b", kind="memory", paths=("b.py",)),
            DriftSubject(id="claim-c", kind="claim", paths=("c.py",)),
        ),
    )


def test_unchanged_is_provenance_only_and_does_not_mutate(drift_repo):
    repo, baseline = drift_repo
    report = analyze_evidence_drift(baseline, repo)
    assert report.snapshot_stable and report.worktree_clean
    assert {s.status for s in report.subjects} == {"inputs_unchanged"}
    assert report.recheck_paths == () and report.file_reads == 3
    assert report.bytes_read == report.read_budget_used_bytes == 30
    assert report.provider_calls == 0
    assert report.disclaimer == "provenance_only_not_semantic_revalidation"


def test_changed_dependency_propagates_but_unrelated_source_not_minimal_seed(
    drift_repo,
):
    repo, baseline = drift_repo
    (repo / "a.py").write_text("VALUE = 2\n")
    commit(repo)
    report = analyze_evidence_drift(baseline, repo)
    statuses = {s.id: s for s in report.subjects}
    assert report.baseline_commit != report.current_commit
    assert report.recheck_paths == ("a.py", "b.py")
    assert report.full_recheck_paths == ("a.py", "b.py", "c.py")
    assert statuses["claim-a"].status == "evidence_invalidated"
    assert statuses["memory-b"].reasons == ("recorded_dependency_changed",)
    assert (
        statuses["claim-c"].status == "revalidation_required"
    )  # not auto-promoted across commits


@pytest.mark.parametrize(
    "condition", ["dirty", "missing", "symlink", "incomplete", "truncated"]
)
def test_uncertainty_never_justifies_narrow_recheck(drift_repo, condition):
    repo, baseline = drift_repo
    if condition == "dirty":
        (repo / "extra.txt").write_text("untracked")
    elif condition == "missing":
        (repo / "a.py").unlink()
        commit(repo)
    elif condition == "symlink":
        (repo / "a.py").unlink()
        (repo / "a.py").symlink_to("b.py")
        commit(repo)
    elif condition == "incomplete":
        baseline = baseline.model_copy(
            update={
                "subjects": (
                    *baseline.subjects,
                    DriftSubject(
                        id="unknown", kind="claim", paths=(), provenance_complete=False
                    ),
                )
            }
        )
    else:
        baseline = baseline.model_copy(update={"scope_truncated": True})
    report = analyze_evidence_drift(baseline, repo)
    assert report.recheck_paths == report.full_recheck_paths
    if condition in ("missing", "symlink"):
        assert report.sources[0].status == "unavailable"
        assert report.sources[0].reason == condition


def test_conflicting_digest_is_not_silently_selected(drift_repo):
    repo, baseline = drift_repo
    baseline = baseline.model_copy(
        update={
            "anchors": (*baseline.anchors, SourceAnchor(path="a.py", digest="a" * 64))
        }
    )
    report = analyze_evidence_drift(baseline, repo)
    assert report.sources[0].status == "conflicting_baseline"
    assert report.sources[0].current_digest is None
    assert report.file_reads == 2
    assert report.recheck_paths == report.full_recheck_paths


def test_failed_read_consumes_bounded_byte_reservation(drift_repo, monkeypatch):
    import repocompass.structure.drift as drift
    from repocompass.memory.revalidation import (
        MemorySourceReadError,
        MemorySourceReadReason,
    )

    repo, baseline = drift_repo
    calls = []

    def fail(*_, max_bytes):
        calls.append(max_bytes)
        raise MemorySourceReadError(MemorySourceReadReason.CHANGED_DURING_READ)

    monkeypatch.setattr(drift, "MAX_DRIFT_BYTES", 12)
    monkeypatch.setattr(drift, "read_memory_source_bytes", fail)
    report = analyze_evidence_drift(baseline, repo)
    assert calls == [11]
    assert report.bytes_read == 0 and report.read_budget_used_bytes == 12
    assert report.scope_truncated and report.file_reads == 1


def test_foreign_repository_rejected_before_read(drift_repo, monkeypatch):
    import repocompass.structure.drift as drift

    repo, baseline = drift_repo

    def forbidden(*_, **kwargs):
        raise AssertionError("must not read foreign source")

    monkeypatch.setattr(drift, "read_memory_source_bytes", forbidden)
    with pytest.raises(ValueError, match="another repository"):
        analyze_evidence_drift(
            baseline.model_copy(update={"repository_id": "f" * 64}), repo
        )


def test_outside_slice_change_forces_full_scoped_recheck(drift_repo):
    repo, baseline = drift_repo
    (repo / "new_dependency.py").write_text("VALUE = 42\n")
    commit(repo)
    report = analyze_evidence_drift(baseline, repo)
    assert report.changes_outside_scope is True
    assert report.recheck_paths == report.full_recheck_paths
    assert {source.status for source in report.sources} == {"unchanged"}
    assert {subject.status for subject in report.subjects} == {"revalidation_required"}
