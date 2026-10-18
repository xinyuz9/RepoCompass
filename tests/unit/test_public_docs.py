"""Keep the public entry points usable in a clean checkout."""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def test_public_documentation_links_resolve():
    documents = [
        ROOT / name
        for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "NOTICE.md")
    ]
    documents.extend((ROOT / "docs").glob("*.md"))
    for document in documents:
        for target in re.findall(r"\]\(([^\s)]+)\)", document.read_text()):
            link = urlsplit(target)
            if not link.scheme and link.path:
                assert (document.parent / unquote(link.path)).is_file(), (
                    document.name,
                    target,
                )


def test_distribution_keeps_project_and_bundled_licenses():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT"
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    notices = (
        ROOT / "src/repocompass/dashboard/static/THIRD_PARTY_LICENSES.md"
    ).read_text()
    for dependency in ("react-dom", "JetBrains", "Manrope", "SIL OPEN FONT LICENSE"):
        assert dependency in notices
