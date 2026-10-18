"""Opt-in tests of an already-built local image; no pulls, credentials or LLMs."""

import http.client
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("REPOCOMPASS_RUN_DOCKER_TESTS") != "1",
    reason="set REPOCOMPASS_RUN_DOCKER_TESTS=1 after building repocompass:local",
)


def docker(*args, check=True):
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def get(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return response.read().decode()


def wait_until_ready(url, *, timeout=90):
    """Wait through Docker's published-port startup race, with a fixed deadline."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            get(url + "/api/v1/system")
            return
        except urllib.error.HTTPError:
            raise
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
        ) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("Container readiness timed out") from error
            time.sleep(0.2)


@contextmanager
def running(command, mounts=()):
    name = f"repocompass-smoke-{uuid4().hex}"
    try:
        docker(
            "run",
            "--detach",
            "--pull=never",
            "--init",
            "--name",
            name,
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
            "--publish",
            "127.0.0.1::8766",
            *mounts,
            "repocompass:local",
            *command,
        )
        info = json.loads(docker("inspect", name))[0]
        port = info["NetworkSettings"]["Ports"]["8766/tcp"][0]["HostPort"]
        url = f"http://127.0.0.1:{port}"
        try:
            wait_until_ready(url)
        except TimeoutError:
            pytest.fail("Container readiness timed out: " + docker("logs", name))
        yield name, url
    finally:
        # Only this test's uniquely named container is removed.
        docker("rm", "--force", name, check=False)


def test_image_demo_serves_built_ui_as_non_root():
    with running(["--demo", "--container-network"]) as (name, url):
        system = json.loads(get(url + "/api/v1/system"))
        assert system["demo_mode"] is True
        assert system["lifecycle_mutations_enabled"] is False
        assert system["memory_mutations_enabled"] is False
        assert json.loads(get(url + "/api/v1/threads"))["threads"]
        html = get(url + "/")
        assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
        assert assets
        for asset in assets:
            assert get(url + asset)
        assert docker("exec", name, "id", "-u") == "10001"
        docker("exec", name, "git", "--version")
        docker("exec", name, "rg", "--version")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                urllib.request.Request(
                    url + "/api/v1/system", headers={"Host": "attacker.example"}
                ),
                timeout=3,
            )
        assert error.value.code == 400


def test_image_readonly_checkout_and_sqlite_volume_survive_recreation(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir(mode=0o755)
    (repo / "example.py").write_text("value = 1\n")
    for args in [
        ["init", "-q"],
        ["add", "example.py"],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    volume = f"repocompass-smoke-{uuid4().hex}"
    docker("volume", "create", volume)
    mounts = [
        "--mount",
        f"type=bind,source={repo},target=/workspace/repository,readonly",
        "--mount",
        f"type=volume,source={volume},target=/data",
    ]
    command = ["--container-network", "--repo-path", "/workspace/repository"]
    try:
        with running(command, mounts) as (name, url):
            system = json.loads(get(url + "/api/v1/system"))
            assert system["commit_sha"] == commit
            assert system["lifecycle_mutations_enabled"] is False
            info = json.loads(docker("inspect", name))[0]
            assert info["HostConfig"]["ReadonlyRootfs"] is True
            assert (
                next(
                    m
                    for m in info["Mounts"]
                    if m["Destination"] == "/workspace/repository"
                )["RW"]
                is False
            )
            docker(
                "exec",
                name,
                "python",
                "-c",
                "import sqlite3; from pathlib import Path; "
                "assert Path('/data/.repocompass/checkpoints.sqlite').is_file(); "
                "db=sqlite3.connect('/data/probe.sqlite'); "
                "db.execute('create table probe (value integer)'); "
                "db.execute('insert into probe values (42)'); db.commit()",
            )
            failed_write = subprocess.run(
                [
                    "docker",
                    "exec",
                    name,
                    "touch",
                    "/workspace/repository/should-not-exist",
                ],
                capture_output=True,
                timeout=10,
            )
            assert failed_write.returncode != 0
        with running(command, mounts) as (name, url):
            assert (
                json.loads(get(url + "/api/v1/system"))["repository_id"]
                == system["repository_id"]
            )
            assert (
                docker(
                    "exec",
                    name,
                    "python",
                    "-c",
                    "import sqlite3; db=sqlite3.connect('/data/probe.sqlite'); "
                    "print(db.execute('select value from probe').fetchone()[0])",
                )
                == "42"
            )
        assert not (repo / "should-not-exist").exists()
    finally:
        # This volume contains fixture data only, never the user's named volume.
        docker("volume", "rm", volume)
