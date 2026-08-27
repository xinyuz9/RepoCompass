"""Bounded, non-executing Python AST index tests."""

import hashlib
import os
from pathlib import Path

import pytest

from repocompass.structure import (
    MAX_PYTHON_SOURCE_BYTES,
    PythonFileIndex,
    PythonImportKind,
    PythonIndexError,
    PythonIndexErrorReason,
    PythonReferenceKind,
    PythonSymbolKind,
    build_python_file_index,
)


REPOSITORY_ID = hashlib.sha256(b"repository").hexdigest()
COMMIT_SHA = "b" * 40


def _build(root: Path, relative_path: str) -> PythonFileIndex:
    return build_python_file_index(
        root,
        Path(relative_path),
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
    )


def test_indexes_nested_symbols_imports_decorators_and_references(
    tmp_path: Path,
) -> None:
    target = tmp_path / "pkg/module.py"
    target.parent.mkdir()
    target.write_text(
        "import os as operating_system\n"
        "from .services import Client as ServiceClient, helper\n"
        "\n"
        "@registry.register\n"
        "class Runner(BaseRunner):\n"
        "    label = operating_system.name\n"
        "\n"
        "    @classmethod\n"
        "    async def run(cls, client: ServiceClient):\n"
        "        return helper(client)\n"
        "\n"
        "def duplicate():\n"
        "    return 1\n"
        "\n"
        "def duplicate():\n"
        "    return 2\n",
        encoding="utf-8",
    )

    index = _build(tmp_path, "pkg/module.py")

    assert index.module_name == "pkg.module"
    assert index.path == "pkg/module.py"
    assert index.repository_id == REPOSITORY_ID
    assert index.commit_sha == COMMIT_SHA
    by_qualified_name = {symbol.qualified_name: symbol for symbol in index.symbols}
    assert by_qualified_name["pkg.module"].kind is PythonSymbolKind.MODULE
    runner = by_qualified_name["pkg.module.Runner"]
    run = by_qualified_name["pkg.module.Runner.run"]
    assert runner.kind is PythonSymbolKind.CLASS
    assert runner.decorators == ("registry.register",)
    assert run.kind is PythonSymbolKind.ASYNC_METHOD
    assert run.decorators == ("classmethod",)
    assert run.parent_symbol_id == runner.symbol_id
    duplicates = [
        symbol
        for symbol in index.symbols
        if symbol.qualified_name == "pkg.module.duplicate"
    ]
    assert len(duplicates) == 2
    assert duplicates[0].symbol_id != duplicates[1].symbol_id

    imports = {
        (item.kind, item.module, item.imported_name): item for item in index.imports
    }
    operating_system = imports[(PythonImportKind.IMPORT, "os", "os")]
    service_client = imports[(PythonImportKind.FROM_IMPORT, "services", "Client")]
    assert operating_system.alias == "operating_system"
    assert operating_system.bound_name == "operating_system"
    assert service_client.level == 1
    assert service_client.alias == "ServiceClient"

    references = {(item.kind, item.name) for item in index.references}
    assert (PythonReferenceKind.NAME_LOAD, "BaseRunner") in references
    assert (PythonReferenceKind.ATTRIBUTE_LOAD, "operating_system.name") in references
    assert (PythonReferenceKind.CALL, "helper") in references


def test_duplicate_import_aliases_keep_distinct_source_identity(tmp_path: Path) -> None:
    (tmp_path / "imports.py").write_text(
        "import os, os\nfrom package import value, value\n",
        encoding="utf-8",
    )

    index = _build(tmp_path, "imports.py")

    assert len(index.imports) == 4
    assert len({record.import_id for record in index.imports}) == 4
    assert [record.location.column_start for record in index.imports] == [
        7,
        11,
        20,
        27,
    ]


def test_index_is_deterministic_and_changes_with_source(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")

    first = _build(tmp_path, "module.py")
    second = _build(tmp_path, "module.py")
    target.write_text("def value():\n    return 2\n", encoding="utf-8")
    changed = _build(tmp_path, "module.py")

    assert first == second
    assert first.index_id == second.index_id
    assert first.index_id != changed.index_id
    assert first.source_digest != changed.source_digest


def test_respects_pep263_source_encoding_without_executing_code(tmp_path: Path) -> None:
    source = (
        b"# coding: latin-1\n"
        b"label = 'caf\xe9'\n"
        b"raise RuntimeError('must never execute')\n"
    )
    (tmp_path / "encoded.py").write_bytes(source)

    index = _build(tmp_path, "encoded.py")

    assert index.source_encoding.lower() == "iso-8859-1"
    assert index.source_digest == hashlib.sha256(source).hexdigest()


@pytest.mark.parametrize(
    ("relative_path", "reason"),
    [
        ("../outside.py", PythonIndexErrorReason.INVALID_PATH),
        ("notes.txt", PythonIndexErrorReason.UNSUPPORTED_FILE),
        ("missing.py", PythonIndexErrorReason.FILE_NOT_FOUND),
    ],
)
def test_rejects_invalid_or_unsupported_targets(
    tmp_path: Path,
    relative_path: str,
    reason: PythonIndexErrorReason,
) -> None:
    (tmp_path.parent / "outside.py").write_text("outside = True", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not python", encoding="utf-8")

    with pytest.raises(PythonIndexError) as raised:
        _build(tmp_path, relative_path)

    assert raised.value.reason is reason


def test_rejects_absolute_path_directory_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "inside.py"
    target.write_text("inside = True", encoding="utf-8")
    (tmp_path / "package.py").mkdir()
    (tmp_path / "inside-link.py").symlink_to(target)

    cases = [
        (target, PythonIndexErrorReason.INVALID_PATH),
        (Path("package.py"), PythonIndexErrorReason.SPECIAL_FILE),
        (Path("inside-link.py"), PythonIndexErrorReason.SYMLINK),
    ]
    for path, reason in cases:
        with pytest.raises(PythonIndexError) as raised:
            build_python_file_index(
                tmp_path,
                path,
                repository_id=REPOSITORY_ID,
                commit_sha=COMMIT_SHA,
            )
        assert raised.value.reason is reason


def test_rejects_syntax_encoding_size_and_ast_depth_limits(tmp_path: Path) -> None:
    (tmp_path / "syntax.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "encoding.py").write_bytes(b"value = '\xff'\n")
    (tmp_path / "large.py").write_bytes(b"#" * (MAX_PYTHON_SOURCE_BYTES + 1))
    (tmp_path / "deep.py").write_text(
        "value = " + "not " * 140 + "flag\n",
        encoding="utf-8",
    )

    cases = [
        ("syntax.py", PythonIndexErrorReason.SYNTAX_ERROR),
        ("encoding.py", PythonIndexErrorReason.ENCODING_ERROR),
        ("large.py", PythonIndexErrorReason.OVERSIZED_FILE),
        ("deep.py", PythonIndexErrorReason.AST_LIMIT),
    ]
    for path, reason in cases:
        with pytest.raises(PythonIndexError) as raised:
            _build(tmp_path, path)
        assert raised.value.reason is reason


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_rejects_special_files_without_opening_them(tmp_path: Path) -> None:
    fifo = tmp_path / "events.py"
    os.mkfifo(fifo)

    with pytest.raises(PythonIndexError) as raised:
        _build(tmp_path, "events.py")

    assert raised.value.reason is PythonIndexErrorReason.SPECIAL_FILE
