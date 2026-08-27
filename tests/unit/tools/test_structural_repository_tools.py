"""Gateway and query tests for bounded Python structural tools."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import repocompass.structure.queries as query_module
import repocompass.tools.gateway as gateway_module
from repocompass.structure import (
    PythonIndexError,
    PythonIndexErrorReason,
    discover_python_files,
)
from repocompass.tools.contracts import ToolDecisionOutcome, ToolDecisionReason
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    create_repository_tool_registry,
)
from repocompass.tools.repository_tools import (
    FindReferencesArguments,
    FindSymbolDefinitionArguments,
    InspectImportsArguments,
    ListSymbolsArguments,
    execute_find_references,
    execute_find_symbol_definition,
    execute_inspect_imports,
    execute_list_symbols,
)

REPOSITORY_ID = "a" * 64
COMMIT_SHA = "b" * 40


def _write_fixture(root: Path) -> None:
    package = root / "pkg"
    package.mkdir()
    (package / "alpha.py").write_text(
        "import os as operating_system\n"
        "from .service import Service\n"
        "\n"
        "class Runner:\n"
        "    def run(self):\n"
        "        return Service(operating_system.name)\n",
        encoding="utf-8",
    )
    (package / "beta.py").write_text(
        "from .alpha import Runner\n\ndef build():\n    return Runner()\n",
        encoding="utf-8",
    )


def _execute_gateway(
    root: Path,
    *,
    tool_name: str,
    arguments: dict[str, object],
    gateway: ToolGateway | None = None,
    audit_log: ToolAuditLog | None = None,
    call_id: str = "call-1",
):
    context = ToolExecutionContext(root)
    request = context.bind_call(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    log = audit_log or ToolAuditLog("structural-run")
    result = (gateway or ToolGateway(create_repository_tool_registry())).execute(
        request,
        context=context,
        available_budget=1,
        audit_log=log,
    )
    return result, log, context


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ListSymbolsArguments, {"relative_path": "bad\x00.py"}),
        (ListSymbolsArguments, {"relative_path": "pkg\\module.py"}),
        (InspectImportsArguments, {"relative_path": "module.txt"}),
        (FindSymbolDefinitionArguments, {"name": "not valid"}),
        (FindReferencesArguments, {"name": "Runner", "max_files": 0}),
        (FindReferencesArguments, {"name": "Runner", "unexpected": True}),
    ],
)
def test_structural_argument_models_are_strict_and_bounded(
    model: type,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_public_query_api_validates_identity_name_and_result_bounds_when_empty(
    tmp_path: Path,
) -> None:
    with pytest.raises(PythonIndexError) as invalid_identity:
        query_module.find_python_symbol_definitions(
            tmp_path,
            "Runner",
            relative_path=None,
            max_files=10,
            max_results=10,
            repository_id="invalid",
            commit_sha=COMMIT_SHA,
        )
    with pytest.raises(ValueError, match="max_results"):
        query_module.find_python_references(
            tmp_path,
            "Runner",
            relative_path=None,
            max_files=10,
            max_results=0,
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    with pytest.raises(ValueError, match="dotted Python identifier"):
        query_module.find_python_symbol_definitions(
            tmp_path,
            "not valid",
            relative_path=None,
            max_files=10,
            max_results=10,
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )

    assert invalid_identity.value.reason is PythonIndexErrorReason.INVALID_IDENTITY


def test_list_symbols_and_imports_return_bounded_content_addressed_records(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    symbols = json.loads(
        execute_list_symbols(
            tmp_path,
            ListSymbolsArguments(relative_path="pkg/alpha.py", max_results=2),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    )
    imports = json.loads(
        execute_inspect_imports(
            tmp_path,
            InspectImportsArguments(relative_path="pkg/alpha.py", max_results=1),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    )

    assert symbols["path"] == "pkg/alpha.py"
    assert symbols["total_results"] == 3
    assert symbols["returned_results"] == 2
    assert symbols["results_truncated"] is True
    assert [record["qualified_name"] for record in symbols["symbols"]] == [
        "pkg.alpha",
        "pkg.alpha.Runner",
    ]
    assert all(
        record["verification_status"] == "verified" for record in symbols["symbols"]
    )
    assert all(
        record["definition"]["repository_id"] == REPOSITORY_ID
        for record in symbols["symbols"]
    )
    assert all(
        record["definition"]["commit_sha"] == COMMIT_SHA
        for record in symbols["symbols"]
    )
    assert "content" not in json.dumps(symbols)

    assert imports["path"] == "pkg/alpha.py"
    assert imports["total_results"] == 2
    assert imports["returned_results"] == 1
    assert imports["results_truncated"] is True
    assert imports["imports"][0]["module"] == "os"
    assert (
        imports["imports"][0]["location"]["source_digest"] == symbols["source_digest"]
    )


def test_repository_queries_are_deterministic_bounded_and_explicitly_syntactic(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("class Runner: pass\n", encoding="utf-8")

    definitions = json.loads(
        execute_find_symbol_definition(
            tmp_path,
            FindSymbolDefinitionArguments(
                name="Runner",
                max_files=10,
                max_results=10,
            ),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    )
    references = json.loads(
        execute_find_references(
            tmp_path,
            FindReferencesArguments(
                name="Runner",
                max_files=10,
                max_results=1,
            ),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    )

    assert definitions["query"] == "Runner"
    assert definitions["match_semantics"] == "exact_simple_or_qualified_definition"
    assert [item["qualified_name"] for item in definitions["definitions"]] == [
        "pkg.alpha.Runner"
    ]
    assert definitions["candidate_files"] == 3
    assert definitions["indexed_files"] == 2
    assert definitions["skipped_by_reason"] == {"syntax_error": 1}
    assert definitions["file_scan_truncated"] is False
    assert definitions["results_truncated"] is False

    assert references["match_semantics"] == "syntactic_exact_or_terminal_component"
    assert references["total_results"] >= 2
    assert references["returned_results"] == 1
    assert references["results_truncated"] is True
    assert references["references"][0]["location"]["path"] == "pkg/beta.py"
    assert references["references"][0]["verification_status"] == "verified"

    repeated = json.loads(
        execute_find_symbol_definition(
            tmp_path,
            FindSymbolDefinitionArguments(name="Runner", max_files=10),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    )
    assert repeated == definitions


def test_repository_query_reports_file_scan_truncation_in_stable_path_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("def target(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def target(): pass\n", encoding="utf-8")

    result = json.loads(
        execute_find_symbol_definition(
            tmp_path,
            FindSymbolDefinitionArguments(
                name="target",
                max_files=1,
                max_results=10,
            ),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )
    )

    assert result["candidate_files"] == 1
    assert result["file_scan_truncated"] is True
    assert [item["definition"]["path"] for item in result["definitions"]] == ["a.py"]


def test_python_discovery_ignores_known_directories_and_all_symlinks(
    tmp_path: Path,
) -> None:
    (tmp_path / "visible.py").write_text("visible = True\n", encoding="utf-8")
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (external / "outside.py").write_text("outside = True\n", encoding="utf-8")
    (tmp_path / "linked").symlink_to(external, target_is_directory=True)
    (tmp_path / "linked.py").symlink_to(external / "outside.py")

    discovery = discover_python_files(tmp_path, max_files=10)

    assert discovery.paths == (Path("visible.py"),)
    assert discovery.truncated is False


def test_python_discovery_fails_closed_at_the_directory_entry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(query_module, "MAX_PYTHON_SCAN_ENTRIES", 2)
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("value = True\n", encoding="utf-8")

    discovery = discover_python_files(tmp_path, max_files=10)

    assert discovery.paths == ()
    assert discovery.truncated is True


def test_repository_query_streams_indexes_instead_of_retaining_the_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(5):
        (tmp_path / f"module_{index}.py").write_text(
            "value = True\n",
            encoding="utf-8",
        )
    active_indexes = 0
    peak_active_indexes = 0

    class FakeIndex:
        symbols: tuple[object, ...] = ()
        references: tuple[object, ...] = ()

        def __init__(self) -> None:
            nonlocal active_indexes, peak_active_indexes
            active_indexes += 1
            peak_active_indexes = max(peak_active_indexes, active_indexes)

        def __del__(self) -> None:
            nonlocal active_indexes
            active_indexes -= 1

    def build_fake_index(*_args, **_kwargs) -> FakeIndex:
        return FakeIndex()

    monkeypatch.setattr(query_module, "_build_index", build_fake_index)

    result = query_module.find_python_symbol_definitions(
        tmp_path,
        "missing",
        relative_path=None,
        max_files=10,
        max_results=10,
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
    )

    assert result["indexed_files"] == 5
    assert active_indexes == 0
    assert peak_active_indexes <= 2


def test_direct_file_query_propagates_static_index_error_without_source(
    tmp_path: Path,
) -> None:
    (tmp_path / "secret.py").write_text(
        "authorization = 'Bearer private'\ndef broken(:\n",
        encoding="utf-8",
    )

    with pytest.raises(PythonIndexError) as raised:
        execute_list_symbols(
            tmp_path,
            ListSymbolsArguments(relative_path="secret.py"),
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
        )

    assert str(raised.value) == "python index failed: syntax_error"
    assert "Bearer" not in str(raised.value)


def test_structural_tools_execute_only_through_gateway_and_bind_snapshot_identity(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    result, audit_log, context = _execute_gateway(
        tmp_path,
        tool_name="find_symbol_definition",
        arguments={"name": "Runner", "max_files": 10, "max_results": 5},
    )
    payload = json.loads(result.content)

    assert result.decision.outcome is ToolDecisionOutcome.ALLOWED
    assert result.decision.reason is ToolDecisionReason.ALLOWED
    assert result.budget_consumed == 1
    assert payload["repository_id"] == context.repository_id
    assert payload["commit_sha"] == context.commit_sha
    assert (
        payload["definitions"][0]["definition"]["repository_id"]
        == context.repository_id
    )
    assert audit_log.records == (result.audit_record,)
    assert "pkg/alpha.py" not in result.audit_record.model_dump_json()


@pytest.mark.parametrize(
    "relative_path",
    ["../outside.py", "/etc/passwd.py", "pkg/../../outside.py"],
)
def test_gateway_rejects_structural_path_escape_before_budget(
    tmp_path: Path,
    relative_path: str,
) -> None:
    result, _, _ = _execute_gateway(
        tmp_path,
        tool_name="find_references",
        arguments={"name": "Runner", "relative_path": relative_path},
    )

    assert result.decision.reason is ToolDecisionReason.PATH_ESCAPE
    assert result.budget_consumed == 0


def test_gateway_sanitizes_structural_parse_failure(tmp_path: Path) -> None:
    (tmp_path / "secret.py").write_text(
        "token = 'super-secret'\ndef broken(:\n",
        encoding="utf-8",
    )

    result, _, _ = _execute_gateway(
        tmp_path,
        tool_name="list_symbols",
        arguments={"relative_path": "secret.py"},
    )

    assert result.decision.reason is ToolDecisionReason.EXECUTION_ERROR
    assert result.budget_consumed == 1
    assert json.loads(result.content) == {"error": {"reason": "execution_error"}}
    assert "super-secret" not in result.content
    assert "secret.py" not in result.audit_record.model_dump_json()


def test_structural_result_cache_reuses_only_the_same_clean_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gateway_module, "_discover_commit_sha", lambda _root: COMMIT_SHA
    )
    monkeypatch.setattr(gateway_module, "_is_clean_git_worktree", lambda _root: True)
    _write_fixture(tmp_path)
    gateway = ToolGateway(create_repository_tool_registry())
    audit_log = ToolAuditLog("structural-run")

    first, _, _ = _execute_gateway(
        tmp_path,
        tool_name="list_symbols",
        arguments={"relative_path": "pkg/alpha.py", "max_results": 10},
        gateway=gateway,
        audit_log=audit_log,
        call_id="call-first",
    )
    second, _, _ = _execute_gateway(
        tmp_path,
        tool_name="list_symbols",
        arguments={"max_results": 10, "relative_path": "pkg/alpha.py"},
        gateway=gateway,
        audit_log=audit_log,
        call_id="call-second",
    )

    assert first.decision.reason is ToolDecisionReason.ALLOWED
    assert second.decision.reason is ToolDecisionReason.CACHE_HIT
    assert second.budget_consumed == 0
    assert second.content == first.content
