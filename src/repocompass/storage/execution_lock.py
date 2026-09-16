"""Local POSIX advisory locks for coordinated request and checkpoint writers."""

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ExecutionLockUnavailable(RuntimeError):
    """An executor is active, or its local ownership cannot be established."""


@contextmanager
def execution_file_lock(
    directory: Path, request_id: str, *, create: bool
) -> Iterator[None]:
    if os.name != "posix":
        raise ExecutionLockUnavailable(
            "local execution recovery requires POSIX locking"
        )
    import fcntl

    if create:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (hashlib.sha256(request_id.encode()).hexdigest() + ".lock")
    descriptor = None
    try:
        descriptor = os.open(
            path, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0), 0o600
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ExecutionLockUnavailable(
            "execution owner is active or unavailable"
        ) from error
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
