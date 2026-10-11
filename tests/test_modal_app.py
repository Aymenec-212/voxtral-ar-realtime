"""The CPU-testable part of deploy/modal_app.py: the startup guard."""

import sys
import time

import pytest

from deploy.modal_app import wait_until_listening
from tests.conftest import free_port

EXITS_3 = [sys.executable, "-c", "raise SystemExit(3)"]


def listener(port: int) -> list[str]:
    code = (
        f"import socket, time; s = socket.socket(); s.bind(('127.0.0.1', {port})); s.listen();"
        "time.sleep(30)"
    )
    return [sys.executable, "-c", code]


def test_returns_once_the_port_accepts_connections():
    import subprocess

    port = free_port()
    process = subprocess.Popen(listener(port))
    try:
        started = time.monotonic()
        wait_until_listening(process, port, poll_s=0.05)
        assert time.monotonic() - started < 10
    finally:
        process.kill()
        process.wait()


def test_raises_with_the_exit_code_when_the_launcher_dies():
    import subprocess

    process = subprocess.Popen(EXITS_3)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 3"):
        wait_until_listening(process, free_port(), poll_s=0.05)
    assert time.monotonic() - started < 10
