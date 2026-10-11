"""Shared fixture: the fake vLLM server running in-process on an ephemeral port."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import uvicorn
from fastapi import FastAPI

from tests.fake_vllm import FakeVLLM


@dataclass
class RunningServer:
    port: int

    @property
    def http(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws(self) -> str:
        return f"ws://127.0.0.1:{self.port}"


@asynccontextmanager
async def serve_app(app: FastAPI, port: int = 0) -> AsyncIterator[RunningServer]:
    """Run an ASGI app with uvicorn inside the current event loop."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()  # surfaces the startup error
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield RunningServer(port)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


def free_port() -> int:
    """A port nothing listens on right now (the OS just handed it out and we released it)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def fake() -> AsyncIterator[FakeVLLM]:
    """A fresh fake vLLM per test; `fake.url` is its realtime WebSocket URL."""
    fake = FakeVLLM()
    async with serve_app(fake.app) as running:
        fake.url = f"{running.ws}/v1/realtime"  # type: ignore[attr-defined]
        fake.health_url = f"{running.http}/health"  # type: ignore[attr-defined]
        yield fake
