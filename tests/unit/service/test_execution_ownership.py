"""Ownership is an OS-held lock plus an exact immutable run binding, not age."""

import sqlite3
import subprocess
import sys

import pytest

from repocompass.service.operation_store import (
    open_service_operation_store,
    ServiceOperationConflictError,
)
from repocompass.storage.execution_lock import ExecutionLockUnavailable


def _reserve(store, request_id="request-1"):
    return store.reserve(
        request_id=request_id,
        request_digest="a" * 64,
        kind="start",
        proposed_thread_id="thread-1",
    )


def test_active_owner_blocks_reconciliation_and_other_requests(tmp_path):
    path = tmp_path / "operations.sqlite"
    with (
        open_service_operation_store(path) as writer,
        open_service_operation_store(path) as observer,
    ):
        _reserve(writer)
        assert observer.recovery_availability("request-1") == "unavailable"
        with writer.execution("request-1"):
            writer.bind_run("request-1", "run-1")
            writer.bind_run("request-1", "run-1")
            assert observer.execution_run("request-1") == "run-1"
            assert (
                observer.recovery_availability("request-1") == "active_or_unavailable"
            )
            with pytest.raises(ExecutionLockUnavailable):
                with observer.recovery("request-1"):
                    pytest.fail("active execution must not be stolen")
            with pytest.raises(ServiceOperationConflictError):
                writer.bind_run("request-1", "run-2")
            with pytest.raises(ServiceOperationConflictError):
                _reserve(observer, "different-request")
        assert observer.recovery_availability("request-1") == "reconcilable"
        with pytest.raises(ServiceOperationConflictError):
            with writer.execution("request-1"):
                pytest.fail("must not restart enrolled execution")


def test_process_death_releases_owner_but_preserves_run_binding(tmp_path):
    path = tmp_path / "operations.sqlite"
    code = """
import os,sys
from pathlib import Path
from repocompass.service.operation_store import open_service_operation_store
with open_service_operation_store(Path(sys.argv[1])) as store:
    store.reserve(request_id='request-1',request_digest='a'*64,kind='start',proposed_thread_id='thread-1')
    with store.execution('request-1'):
        store.bind_run('request-1','run-1')
        os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, timeout=15
    )
    assert result.returncode == 17, result.stderr.decode()
    with open_service_operation_store(path) as store:
        assert store.get("request-1").state.value == "pending"
        with store.recovery("request-1") as run_id:
            assert run_id == "run-1"
        assert store.get("request-1").state.value == "pending"


def test_legacy_missing_lock_and_symlink_fail_closed(tmp_path):
    path = tmp_path / "operations.sqlite"
    with open_service_operation_store(path) as store:
        _reserve(store)
        with pytest.raises(ExecutionLockUnavailable):
            with store.recovery("request-1"):
                pytest.fail("legacy must not become recoverable")
        with store.execution("request-1"):
            store.bind_run("request-1", "run-1")
        lock = next((tmp_path / "operations.sqlite.locks").glob("*.lock"))
        lock.unlink()
        with pytest.raises(ExecutionLockUnavailable):
            with store.recovery("request-1"):
                pytest.fail("missing lock is not proof of inactivity")
        lock.symlink_to(path)
        assert store.recovery_availability("request-1") == "active_or_unavailable"
    with sqlite3.connect(path) as c:
        assert (
            c.execute("SELECT run_id FROM service_executions").fetchone()[0] == "run-1"
        )
