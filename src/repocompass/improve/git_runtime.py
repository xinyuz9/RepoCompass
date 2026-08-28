"""Constrained process configuration for focused-improve Git operations."""

from __future__ import annotations

import os


def safe_git_command(*arguments: str) -> tuple[str, ...]:
    """Return Git argv with host-executing integration points disabled."""

    if not all(
        isinstance(argument, str) and "\x00" not in argument for argument in arguments
    ):
        raise TypeError("Git arguments must be NUL-free strings")
    return (
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "credential.helper=",
        *arguments,
    )


def safe_git_environment() -> dict[str, str]:
    """Build a minimal cross-platform environment without user Git config."""

    environment = {
        name: value
        for name in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP")
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    return environment
