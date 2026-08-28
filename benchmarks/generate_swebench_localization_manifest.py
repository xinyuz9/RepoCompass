"""Build a frozen SWE-bench Lite localization manifest.

The input is an official Hugging Face datasets-server response. Gold patch
bodies are consumed only in memory and are never written to the manifest.
Network access for base-commit source lookup requires an explicit flag.
"""

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from repocompass.evaluation import (
    LocalizationDatasetKind,
    LocalizationLabelSource,
    create_localization_case,
    create_localization_dataset_manifest,
    create_localization_labels,
    create_localization_task,
    derive_gold_patch_localization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "benchmarks/cases/swe-bench-lite-localization-10-v1.json"
)
SCALE_OUTPUT = PROJECT_ROOT / "benchmarks/cases/swe-bench-lite-localization-40-v1.json"
UPSTREAM_DATASET = "SWE-bench/SWE-bench_Lite"
UPSTREAM_REVISION = "b0dde1093fe417d83b7184254edf8199c1f0dff5"
UPSTREAM_RESPONSE_SHA256 = (
    "b8bbed5cd0d76b3ef16bb8801a169be209855a85bffb433be9c970ba7878fce9"
)
MANIFEST_VERSION = "hf-b0dde1093fe4-v1"
MAX_UPSTREAM_RESPONSE_BYTES = 16_777_216
MAX_BASE_SOURCE_BYTES = 1_048_576
EXPECTED_CASE_COUNT = 10

SourceLoader = Callable[[str, str, str], str | None]


@dataclass(frozen=True, slots=True)
class FrozenSourceProfile:
    """One reviewed datasets-server response and its manifest destination."""

    case_count: int
    response_sha256: str
    manifest_name: str
    output_path: Path


FROZEN_SOURCE_PROFILES = {
    10: FrozenSourceProfile(
        case_count=10,
        response_sha256=UPSTREAM_RESPONSE_SHA256,
        manifest_name="swe-bench-lite-localization-10",
        output_path=DEFAULT_OUTPUT,
    ),
    40: FrozenSourceProfile(
        case_count=40,
        response_sha256=(
            "24f401102bd0ab4e946f447106c873a9ff645317bdb8161782f59b8820da377e"
        ),
        manifest_name="swe-bench-lite-localization-40",
        output_path=SCALE_OUTPUT,
    ),
}


def load_upstream_rows_file(
    path: Path,
    *,
    expected_sha256: str = UPSTREAM_RESPONSE_SHA256,
    expected_count: int = EXPECTED_CASE_COUNT,
) -> tuple[dict[str, str], ...]:
    """Load a bounded duplicate-key-free official rows response."""

    if not isinstance(path, Path):
        raise TypeError("upstream rows path must be a Path")
    raw = path.read_bytes()
    if len(raw) > MAX_UPSTREAM_RESPONSE_BYTES:
        raise ValueError(
            f"upstream rows response exceeds {MAX_UPSTREAM_RESPONSE_BYTES} bytes"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("upstream rows response digest does not match frozen source")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"invalid upstream rows response: {error}") from error
    return normalize_upstream_rows(payload, expected_count=expected_count)


def normalize_upstream_rows(
    payload: object,
    *,
    expected_count: int,
) -> tuple[dict[str, str], ...]:
    """Extract only task and gold-patch fields from an untrusted API payload."""

    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise TypeError("expected_count must be an integer")
    if expected_count < 1 or expected_count > 1_000:
        raise ValueError("expected_count must be between 1 and 1000")
    if not isinstance(payload, Mapping):
        raise ValueError("upstream rows response must be a JSON object")
    row_envelopes = payload.get("rows")
    if not isinstance(row_envelopes, list) or len(row_envelopes) != expected_count:
        raise ValueError("upstream rows response has an unexpected case count")
    normalized = []
    required = ("instance_id", "repo", "base_commit", "problem_statement", "patch")
    for index, envelope in enumerate(row_envelopes):
        if not isinstance(envelope, Mapping):
            raise ValueError(f"upstream row envelope {index} must be an object")
        row = envelope.get("row")
        if not isinstance(row, Mapping):
            raise ValueError(f"upstream row {index} must contain an object")
        values: dict[str, str] = {}
        for field in required:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"upstream row {index} has invalid {field}")
            values[field] = value
        normalized.append(values)
    instance_ids = [row["instance_id"] for row in normalized]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("upstream rows contain duplicate instance IDs")
    return tuple(sorted(normalized, key=lambda row: row["instance_id"]))


def build_manifest(
    rows: Sequence[Mapping[str, str]],
    *,
    source_loader: SourceLoader,
    name: str = "swe-bench-lite-localization-10",
    version: str = MANIFEST_VERSION,
):
    """Derive a label-only manifest from normalized official rows."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of normalized mappings")
    if not callable(source_loader):
        raise TypeError("source_loader must be callable")
    cases = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each normalized row must be a mapping")
        repository = row["repo"]
        base_commit = row["base_commit"]
        task = create_localization_task(
            dataset_kind=LocalizationDatasetKind.SWE_BENCH_LITE_DERIVED,
            instance_id=row["instance_id"],
            repository=repository,
            base_commit=base_commit,
            objective=row["problem_statement"],
        )
        initial = derive_gold_patch_localization(row["patch"], base_sources={})
        sources: dict[str, str] = {}
        for path in initial.files:
            if not path.endswith(".py"):
                continue
            source = source_loader(repository, base_commit, path)
            if source is not None:
                sources[path] = source
        derivation = derive_gold_patch_localization(
            row["patch"],
            base_sources=sources,
        )
        labels = create_localization_labels(
            task_id=task.task_id,
            source=LocalizationLabelSource.SWE_BENCH_GOLD_PATCH,
            source_artifact_digest=derivation.source_artifact_digest,
            gold_files=derivation.files,
            gold_symbols=derivation.symbols,
        )
        cases.append(create_localization_case(task=task, labels=labels))
    return create_localization_dataset_manifest(
        name=name,
        version=version,
        kind=LocalizationDatasetKind.SWE_BENCH_LITE_DERIVED,
        cases=tuple(cases),
    )


class BaseSourceLoader:
    """Explicitly network-gated, bounded raw GitHub source cache."""

    def __init__(self, cache_root: Path, *, allow_network: bool) -> None:
        if not isinstance(cache_root, Path):
            raise TypeError("source cache root must be a Path")
        self._cache_root = cache_root.resolve()
        self._allow_network = allow_network

    def __call__(self, repository: str, commit: str, path: str) -> str | None:
        cache_path = self._cache_root / repository.replace("/", "__") / commit / path
        try:
            cache_path.resolve().relative_to(self._cache_root)
        except ValueError as error:
            raise ValueError("source cache path escapes its trusted root") from error
        if cache_path.is_file():
            return _decode_bounded_source(cache_path.read_bytes(), path=path)
        if not self._allow_network:
            return None
        quoted_path = urllib.parse.quote(path, safe="/")
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{quoted_path}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "RepoCompass-localization-fixture/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read(MAX_BASE_SOURCE_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise ValueError(
                f"base source request failed with HTTP {error.code}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ValueError("base source request failed") from error
        source = _decode_bounded_source(raw, path=path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.write_bytes(raw)
        except OSError as error:
            raise ValueError(
                f"could not persist bounded source cache: {path}"
            ) from error
        return source


def _decode_bounded_source(raw: bytes, *, path: str) -> str:
    if len(raw) > MAX_BASE_SOURCE_BYTES:
        raise ValueError(f"base source exceeds {MAX_BASE_SOURCE_BYTES} bytes: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"base source is not UTF-8: {path}") from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument(
        "--case-count",
        type=int,
        choices=tuple(sorted(FROZEN_SOURCE_PROFILES)),
        default=EXPECTED_CASE_COUNT,
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    profile = FROZEN_SOURCE_PROFILES[arguments.case_count]
    rows = load_upstream_rows_file(
        arguments.source_file,
        expected_sha256=profile.response_sha256,
        expected_count=profile.case_count,
    )
    manifest = build_manifest(
        rows,
        source_loader=BaseSourceLoader(
            arguments.source_cache,
            allow_network=arguments.allow_network,
        ),
        name=profile.manifest_name,
    )
    output_path = arguments.output or profile.output_path
    mode = "w" if arguments.force else "x"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(manifest.model_dump_json(indent=2, exclude_computed_fields=True))
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
