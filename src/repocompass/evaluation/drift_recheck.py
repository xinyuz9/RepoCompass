"""Offline paired re-index experiment, not a model or runtime-semantic benchmark.

All writes are to a new TemporaryDirectory. No target repository code is imported.
Hand-labelled invalidation obligations are separate from production drift decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

from repocompass.structure import python_index
from repocompass.structure import drift
from repocompass.tools.gateway import ToolExecutionContext

CASES = (
    "unchanged",
    "renamed_symbol",
    "syntax_error",
    "deleted_source",
    "symlink_source",
    "dirty_outside",
    "committed_outside",
    "incomplete",
    "truncated",
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, timeout=10
    )


def _commit(root: Path) -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Drift fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-qm",
        "fixture",
    )


def _fixture(root: Path, case: str, *, file_count: int, functions: int):
    root.mkdir()
    _git(root, "init", "-q")
    sources = {
        "policy.py": "def verify(value):\n    return bool(value)\n",
        "api.py": "from policy import verify\n\ndef apply(value):\n    return verify(value)\n",
    }
    for file in range(file_count - 2):
        sources[f"utility_{file:02d}.py"] = "\n".join(
            f"def helper_{index}(value):\n    return value + {index}\n"
            for index in range(functions)
        )
    for path, source in sources.items():
        (root / path).write_text(source, encoding="utf-8")
    _commit(root)
    identity = ToolExecutionContext(root)
    baseline = drift.DriftBaseline(
        repository_id=identity.repository_id,
        commit_sha=identity.commit_sha,
        anchors=tuple(
            drift.SourceAnchor(
                path=path, digest=hashlib.sha256(source.encode()).hexdigest()
            )
            for path, source in sources.items()
        ),
        subjects=tuple(
            drift.DriftSubject(id=path, kind="entity", paths=(path,))
            for path in sources
        ),
        dependencies=(("policy.py", "api.py"),),
    )
    return baseline


def _mutate(root: Path, baseline: drift.DriftBaseline, case: str):
    """Predeclared labels, never computed from drift output or the indexer."""
    all_paths = {anchor.path for anchor in baseline.anchors}
    required = set()
    error = None
    if case == "renamed_symbol":
        (root / "policy.py").write_text(
            "def validate(value):\n    return bool(value)\n"
        )
        required = {"policy.py", "api.py"}
    elif case == "syntax_error":
        (root / "policy.py").write_text("def verify(:\n")
        required, error = {"policy.py", "api.py"}, "syntax_error"
    elif case == "deleted_source":
        (root / "policy.py").unlink()
        required, error = all_paths, "file_not_found"
    elif case == "symlink_source":
        (root / "policy.py").unlink()
        (root / "policy.py").symlink_to("api.py")
        required, error = all_paths, "symlink"
    elif case in {"dirty_outside", "committed_outside"}:
        (root / "new_dependency.py").write_text("UNSEEN = True\n")
        required = all_paths
    elif case == "incomplete":
        baseline = baseline.model_copy(
            update={
                "subjects": (
                    *baseline.subjects,
                    drift.DriftSubject(
                        id="unknown", kind="claim", paths=(), provenance_complete=False
                    ),
                )
            }
        )
        required = all_paths
    elif case == "truncated":
        baseline = baseline.model_copy(update={"scope_truncated": True})
        required = all_paths
    if case in {
        "renamed_symbol",
        "syntax_error",
        "deleted_source",
        "symlink_source",
        "committed_outside",
    }:
        _commit(root)
    return baseline, required, error


def _index(root: Path, paths, identity):
    records = {}
    for path in sorted(paths):
        try:
            index = python_index.build_python_file_index(
                root,
                Path(path),
                repository_id=identity.repository_id,
                commit_sha=identity.commit_sha,
            )
            # Snapshot-independent comparison only. Never mint a new content-addressed
            # index from old records, promote memory, or claim execution equivalence.
            records[path] = {
                "digest": index.source_digest,
                "symbols": sorted(
                    (
                        s.qualified_name,
                        s.kind.value,
                        s.definition.line_start,
                        s.definition.line_end,
                    )
                    for s in index.symbols
                ),
                "imports": sorted(
                    (
                        i.module or "",
                        i.imported_name or "",
                        i.bound_name or "",
                        i.level,
                        i.location.line_start,
                    )
                    for i in index.imports
                ),
                "references": sorted(
                    (r.name, r.kind.value, r.location.line_start)
                    for r in index.references
                ),
            }
        except python_index.PythonIndexError as error:
            records[path] = {"error": error.reason.value}
    return records


def _measure(action):
    """Instrument successfully returned source bytes (not physical disk I/O).

    Reader failures are counted separately; their unknown partial bytes are never zero.
    Git metadata reads, Python imports and the OS cache are outside this byte metric,
    but inside measured elapsed time where they occur in the action.
    """
    counter = {"source_read_attempts": 0, "source_bytes_returned": 0, "failed_reads": 0}

    def wrapped(reader):
        def read(*args, **kwargs):
            counter["source_read_attempts"] += 1
            try:
                content = reader(*args, **kwargs)
            except Exception:
                counter["failed_reads"] += 1
                raise
            counter["source_bytes_returned"] += len(content)
            return content

        return read

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                drift,
                "read_memory_source_bytes",
                wrapped(drift.read_memory_source_bytes),
            )
        )
        stack.enter_context(
            patch.object(
                python_index,
                "_read_bounded_regular_file",
                wrapped(python_index._read_bounded_regular_file),
            )
        )
        start = perf_counter_ns()
        result = action()
        elapsed = perf_counter_ns() - start
    return result, {**counter, "elapsed_ns": elapsed}


def run_case(
    case: str, *, repeats: int = 5, file_count: int = 24, functions: int = 80
) -> dict:
    if (
        case not in CASES
        or not 1 <= repeats <= 20
        or not 3 <= file_count <= 64
        or not 1 <= functions <= 100
    ):
        raise ValueError("invalid bounded experiment configuration")
    with tempfile.TemporaryDirectory(prefix="repocompass-drift-") as directory:
        root = Path(directory) / "fixture"
        baseline = _fixture(root, case, file_count=file_count, functions=functions)
        paths = [anchor.path for anchor in baseline.anchors]
        cached, setup_cost = _measure(
            lambda: _index(root, paths, ToolExecutionContext(root))
        )
        baseline, required, expected_error = _mutate(root, baseline, case)
        samples = {"full": [], "selective": []}
        equivalent = independent_pass = True
        selected_paths = []
        for repetition in range(repeats):
            outputs = {}

            def full():
                return _index(root, paths, ToolExecutionContext(root)), paths

            def selective():
                report = drift.analyze_evidence_drift(baseline, root)
                changed = _index(root, report.recheck_paths, ToolExecutionContext(root))
                return cached | changed, list(report.recheck_paths)

            for name in (
                ("full", "selective") if repetition % 2 == 0 else ("selective", "full")
            ):
                (outputs[name], selected), cost = _measure(
                    full if name == "full" else selective
                )
                samples[name].append({**cost, "indexed_files": len(selected)})
                if name == "selective":
                    selected_paths = sorted(selected)
            equivalent &= outputs["full"] == outputs["selective"]
            independent_pass &= required <= set(selected_paths)
            record = outputs["selective"]["policy.py"]
            if expected_error:
                independent_pass &= record == {"error": expected_error}
            else:
                expected_symbol = (
                    "policy.validate" if case == "renamed_symbol" else "policy.verify"
                )
                independent_pass &= expected_symbol in {
                    symbol[0] for symbol in record["symbols"]
                }
        return {
            "case": case,
            "full_scoped_files": len(paths),
            "recheck_paths": selected_paths,
            "expected_required_paths": sorted(required),
            "structural_projection_equal": equivalent,
            "hand_labelled_obligations_pass": independent_pass,
            "baseline_setup": setup_cost,
            "samples": samples,
            "median_elapsed_ms": {
                name: statistics.median(s["elapsed_ns"] for s in values) / 1e6
                for name, values in samples.items()
            },
            "semantic_equivalence": "not_measured",
            "provider_calls": 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; preserve previous experiments")
    report = {
        "schema_version": 1,
        "provenance": "synthetic_static_reindex_not_runtime_or_llm_quality",
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "implementation_digests": {
            module.__name__: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (drift, python_index)
        },
        "harness_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "cases": [run_case(case, repeats=args.repeats) for case in CASES],
    }
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(
        json.dumps(
            {
                "report_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "output": str(args.output),
                "provider_calls": 0,
            }
        )
    )
    return (
        0
        if all(
            c["structural_projection_equal"] and c["hand_labelled_obligations_pass"]
            for c in report["cases"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
