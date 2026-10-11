"""`python -m voxtral_rt` with UPSTREAM_URL set: no weights, no engine, just the gateway."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

from tests.conftest import free_port

ROOT = Path(__file__).resolve().parents[1]


async def test_entry_point_serves_the_gateway_against_upstream_url(fake):
    port = free_port()
    env = {**os.environ, "UPSTREAM_URL": fake.url, "PORT": str(port), "MODELS_DIR": "/nonexistent"}
    process = subprocess.Popen(
        [sys.executable, "-m", "voxtral_rt"], env=env, cwd=ROOT, stderr=subprocess.PIPE, text=True
    )
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
            async with asyncio.timeout(15):
                while True:
                    try:
                        if (await http.get("/health")).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert process.poll() is None, process.stderr.read()
                    await asyncio.sleep(0.1)
        async with connect(f"ws://127.0.0.1:{port}/v1/transcribe") as ws:
            assert json.loads(await ws.recv()) == {"type": "ready"}
            for _ in range(3):
                await ws.send(b"\0\0" * 1600)
            await ws.send(json.dumps({"type": "stop"}))
            events = [json.loads(await ws.recv()) for _ in range(2)]
        assert [e["type"] for e in events] == ["delta", "final"]
    finally:
        process.terminate()
        process.wait(timeout=10)
