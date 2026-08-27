"""Local-only repository identity resolution for commit-aware memory."""

import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from repocompass.memory.contracts import (
    MemoryRepositoryIdentity,
    MemoryRepositoryIdentityKind,
    create_memory_repository_identity,
)
from repocompass.tools.gateway import ToolExecutionContext

_GIT_IDENTITY_TIMEOUT_SECONDS = 2
_SCP_REMOTE_PATTERN = re.compile(
    r"^(?:[^@/:\s]+@)?(?P<host>[^:/\s]+):(?P<path>[^\s]+)$"
)


def resolve_memory_repository_identity(
    context: ToolExecutionContext,
) -> MemoryRepositoryIdentity:
    """Resolve a credential-free clone grouping key without network access."""

    if not isinstance(context, ToolExecutionContext):
        raise TypeError("context must be a ToolExecutionContext")

    top_level_text = _git_output(
        context.repository_root,
        ("rev-parse", "--show-toplevel"),
    )
    remote_text = _git_output(
        context.repository_root,
        ("config", "--get", "remote.origin.url"),
    )
    if top_level_text is not None and remote_text is not None:
        try:
            top_level = Path(top_level_text).resolve(strict=True)
            relative_scope = context.repository_root.relative_to(top_level)
        except (OSError, ValueError):
            pass
        else:
            canonical_locator = _canonical_remote_locator(
                remote_text,
                top_level=top_level,
            )
            if canonical_locator is not None:
                scope = (
                    "."
                    if not relative_scope.parts
                    else PurePosixPath(*relative_scope.parts).as_posix()
                )
                return create_memory_repository_identity(
                    kind=MemoryRepositoryIdentityKind.GIT_REMOTE,
                    locator_digest=_locator_digest(canonical_locator),
                    scope=scope,
                    source_repository_id=context.repository_id,
                )

    return create_memory_repository_identity(
        kind=MemoryRepositoryIdentityKind.LOCAL_PATH,
        locator_digest=context.repository_id,
        scope=".",
        source_repository_id=context.repository_id,
    )


def _git_output(repository_root: Path, arguments: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=_GIT_IDENTITY_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output or "\x00" in output or "\n" in output or "\r" in output:
        return None
    return output


def _canonical_remote_locator(remote: str, *, top_level: Path) -> str | None:
    """Return a secret-free canonical locator, or fail to clone-local identity."""

    value = remote.strip()
    if not value or "\x00" in value or any(character.isspace() for character in value):
        return None

    scp_match = _SCP_REMOTE_PATTERN.fullmatch(value)
    if scp_match is not None and "://" not in value:
        return _canonical_host_path(
            scp_match.group("host"),
            scp_match.group("path"),
            port=None,
        )

    if "://" in value:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if parsed.query or parsed.fragment:
            return None
        if scheme == "file":
            decoded = unquote(parsed.path)
            if not decoded:
                return None
            try:
                resolved = Path(decoded).expanduser().resolve(strict=False)
            except OSError:
                return None
            return f"local/{resolved.as_posix()}"
        if scheme not in {"git", "http", "https", "ssh"}:
            return None
        if parsed.hostname is None:
            return None
        try:
            port = parsed.port
        except ValueError:
            return None
        return _canonical_host_path(
            parsed.hostname,
            unquote(parsed.path),
            port=port,
        )

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = top_level / path
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return None
    return f"local/{resolved.as_posix()}"


def _canonical_host_path(host: str, path: str, *, port: int | None) -> str | None:
    normalized_host = host.lower().rstrip(".")
    if (
        not normalized_host
        or "\x00" in normalized_host
        or "/" in normalized_host
        or "\\" in normalized_host
    ):
        return None
    normalized_path = path.replace("\\", "/").strip("/")
    if normalized_path.endswith(".git"):
        normalized_path = normalized_path[:-4]
    normalized_path = normalized_path.rstrip("/")
    path_parts = tuple(part for part in normalized_path.split("/") if part)
    if not path_parts or any(
        part in {".", ".."} or "\x00" in part for part in path_parts
    ):
        return None
    authority = normalized_host if port is None else f"{normalized_host}:{port}"
    return f"{authority}/{'/'.join(path_parts)}"


def _locator_digest(canonical_locator: str) -> str:
    return hashlib.sha256(
        b"repocompass:memory-locator:v1\0" + canonical_locator.encode("utf-8")
    ).hexdigest()
