"""`python -m voxtral_rt` as a real process against the fake vLLM: no weights, no engine."""

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

from tests.conftest import free_port

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "k3y-0123456789abcdef-s3cret"


class Launched:
    def __init__(self, port: int) -> None:
        self.port = port
        self.output = ""  # stdout + stderr, available once the process has exited


@asynccontextmanager
async def launch(fake, **env: str) -> AsyncIterator[Launched]:
    """Run the launcher until the block ends, then collect everything it printed."""
    launched = Launched(free_port())
    # The parent's API_KEY must not leak into the keyless case.
    full_env = {name: value for name, value in os.environ.items() if name != "API_KEY"}
    full_env |= {
        "UPSTREAM_URL": fake.url,
        "PORT": str(launched.port),
        "MODELS_DIR": "/nonexistent",
        "PYTHONUNBUFFERED": "1",
        **env,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "voxtral_rt"],
        env=full_env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{launched.port}") as http:
            async with asyncio.timeout(15):
                while True:
                    assert process.poll() is None, process.stdout.read()
                    try:
                        if (await http.get("/health")).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.1)
        yield launched
    finally:
        process.terminate()
        launched.output, _ = await asyncio.to_thread(process.communicate, timeout=10)


async def one_session(url: str) -> list[dict]:
    async with connect(url) as ws:
        events = [json.loads(await ws.recv())]
        for _ in range(3):
            await ws.send(b"\0\0" * 1600)
        await ws.send(json.dumps({"type": "stop"}))
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] != "metrics":
                events.append(event)
    return events


async def test_keyless_launcher_serves_the_gateway_and_warns(fake):
    async with launch(fake) as server:
        events = await one_session(f"ws://127.0.0.1:{server.port}/v1/transcribe")
    assert [e["type"] for e in events] == ["ready", "delta", "final"]
    assert "WARNING API_KEY is not set: authentication is disabled" in server.output


async def test_the_api_key_never_reaches_the_logs(fake):
    async with launch(fake, API_KEY=API_KEY) as server:
        base = f"127.0.0.1:{server.port}"
        events = await one_session(f"ws://{base}/v1/transcribe?token={API_KEY}")
        assert events[-1]["type"] == "final"
        async with connect(f"ws://{base}/v1/transcribe?token=wrong-guess") as ws:
            await ws.wait_closed()
            assert ws.close_code == 1008
        async with httpx.AsyncClient(base_url=f"http://{base}", timeout=5) as http:
            async with http.stream("GET", f"/v1/metrics/stream?token={API_KEY}") as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        break
            assert (await http.get("/v1/metrics/stream?token=wrong-guess")).status_code == 401

    output = server.output
    assert API_KEY not in output
    assert "wrong-guess" not in output  # a rejected guess is a credential too
    # uvicorn.error's WebSocket line and uvicorn.access's HTTP line, both redacted.
    assert '"WebSocket /v1/transcribe?token=***" [accepted]' in output
    assert '"GET /v1/metrics/stream?token=*** HTTP/1.1" 200' in output
    assert '"GET /v1/metrics/stream?token=*** HTTP/1.1" 401' in output
    assert "authentication is disabled" not in output
