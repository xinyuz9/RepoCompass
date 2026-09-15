"""Verified structural-tool integration at the Active Context boundary."""

import hashlib
import json
from pathlib import Path

from langchain_core.messages import ToolMessage

from repocompass.agent.context import (
    ContextSourceKind,
    RawArtifactKind,
    VerificationStatus,
)
from repocompass.agent.context_gathering import (
    gather_candidate_context,
    has_selected_structural_records,
)
from repocompass.agent.planning import PlanStep
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.tools.repository_tools import (
    ListSymbolsArguments,
    ReadCodeArguments,
    execute_list_symbols,
    execute_read_code,
)


def _step() -> PlanStep:
    return PlanStep(
        id="inspect-structure",
        objective="Inspect relevant symbols and relations.",
        expected_evidence=["AST-backed source locations."],
        status="active",
    )


def test_normalized_read_code_gateway_output_is_verified_source_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    content = execute_read_code(
        tmp_path,
        ReadCodeArguments(
            relative_path="auth.py",
            start_line=1,
            end_line=2,
        ),
    )

    gathered = gather_candidate_context(
        repo_path=tmp_path,
        objective="Explain authentication.",
        active_step=_step(),
        step_stop_reason="none",
        messages=(
            ToolMessage(
                content=content,
                tool_call_id="call-json-read",
                name="read_code",
                status="success",
            ),
        ),
        budget_tokens=8_192,
    )

    evidence = next(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.SOURCE_EXCERPT
    )
    assert evidence.verification_status is VerificationStatus.VERIFIED
    assert evidence.source.location is not None
    assert evidence.source.location.path == "auth.py"
    assert evidence.source.location.line_start == 1
    assert evidence.source.location.line_end == 2
    assert "def authenticate" in evidence.content
    assert not has_selected_structural_records(gathered.pack)


def test_structural_tool_records_become_individual_verified_context_items(
    tmp_path: Path,
) -> None:
    target = tmp_path / "pkg/service.py"
    target.parent.mkdir()
    target.write_text(
        "class Service:\n    def run(self):\n        return True\n",
        encoding="utf-8",
    )
    repository = ToolExecutionContext(tmp_path)
    content = execute_list_symbols(
        tmp_path,
        ListSymbolsArguments(relative_path="pkg/service.py", max_results=10),
        repository_id=repository.repository_id,
        commit_sha=repository.commit_sha,
    )

    gathered = gather_candidate_context(
        repo_path=tmp_path,
        objective="Locate Service.run.",
        active_step=_step(),
        step_stop_reason="none",
        messages=(
            ToolMessage(
                content=content,
                tool_call_id="call-symbols",
                name="list_symbols",
                status="success",
            ),
        ),
        budget_tokens=16_384,
    )
    structural_items = tuple(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.SOURCE_EXCERPT
    )

    assert len(structural_items) == 3
    assert has_selected_structural_records(gathered.pack)
    assert all(
        item.verification_status is VerificationStatus.VERIFIED
        for item in structural_items
    )
    assert {
        item.source.location.symbol
        for item in structural_items
        if item.source.location is not None
    } == {"pkg.service", "pkg.service.Service", "pkg.service.Service.run"}
    assert all(
        item.source.location is not None
        and item.source.location.path == "pkg/service.py"
        for item in structural_items
    )
    source_artifacts = tuple(
        artifact
        for artifact in gathered.raw_artifacts
        if artifact.metadata.kind is RawArtifactKind.SOURCE_EXCERPT
    )
    assert len(source_artifacts) == 1
    assert source_artifacts[0].content.decode("utf-8") == content


def test_forged_or_wrong_snapshot_structural_output_never_becomes_verified(
    tmp_path: Path,
) -> None:
    target = tmp_path / "service.py"
    target.write_text("class Service: pass\n", encoding="utf-8")
    repository = ToolExecutionContext(tmp_path)
    payload = json.loads(
        execute_list_symbols(
            tmp_path,
            ListSymbolsArguments(relative_path="service.py", max_results=10),
            repository_id=repository.repository_id,
            commit_sha=repository.commit_sha,
        )
    )
    payload["repository_id"] = hashlib.sha256(b"wrong-snapshot").hexdigest()
    forged = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    gathered = gather_candidate_context(
        repo_path=tmp_path,
        objective="Inspect Service.",
        active_step=_step(),
        step_stop_reason="none",
        messages=(
            ToolMessage(
                content=forged,
                tool_call_id="call-forged-symbols",
                name="list_symbols",
                status="success",
            ),
        ),
        budget_tokens=8_192,
    )

    tool_items = tuple(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.TOOL_RESULT
    )
    assert len(tool_items) == 1
    assert tool_items[0].verification_status is VerificationStatus.UNVERIFIED
    assert not has_selected_structural_records(gathered.pack)
    assert not any(
        item.source.kind is ContextSourceKind.SOURCE_EXCERPT
        for item in gathered.pack.items
    )
