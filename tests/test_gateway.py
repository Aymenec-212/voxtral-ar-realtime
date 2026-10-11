import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from tests.conftest import free_port, serve_app
from tests.fake_vllm import FakeVLLM
from voxtral_rt.config import Settings
from voxtral_rt.gateway import create_app

PCM = bytes(range(256)) * 12


@asynccontextmanager
async def gateway(fake, **overrides):
    settings = replace(Settings.from_env({}), upstream_url=fake.url, **overrides)
    async with serve_app(create_app(settings)) as running:
        yield running


async def recv(ws) -> dict:
    return json.loads(await ws.recv())


async def drain(ws) -> tuple[list[dict], int]:
    """Collect every remaining message and the close code."""
    events = []
    try:
        while True:
            events.append(await recv(ws))
    except ConnectionClosed as exc:
        return events, exc.rcvd.code if exc.rcvd else -1


async def wait_until(predicate, timeout=3.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_happy_path(fake):
    async with gateway(fake) as gw, connect(f"{gw.ws}/v1/transcribe") as ws:
        assert await recv(ws) == {"type": "ready"}
        for _ in range(6):
            await ws.send(PCM)
        await ws.send(json.dumps({"type": "stop"}))
        events, close_code = await drain(ws)

    assert [e["type"] for e in events] == ["delta", "delta", "final"]
    assert events[-1]["text"] == "".join(e["text"] for e in events[:-1])
    assert close_code == 1000


async def test_auth_off_ignores_token(fake):
    async with gateway(fake) as gw, connect(f"{gw.ws}/v1/transcribe?token=whatever") as ws:
        assert await recv(ws) == {"type": "ready"}


async def test_auth_on_rejects_wrong_or_missing_token(fake):
    async with gateway(fake, api_key="s3cret") as gw:
        for url in (f"{gw.ws}/v1/transcribe", f"{gw.ws}/v1/transcribe?token=nope"):
            async with connect(url) as ws:
                events, close_code = await drain(ws)
            assert events == []
            assert close_code == 1008
        async with connect(f"{gw.ws}/v1/transcribe?token=s3cret") as ws:
            assert await recv(ws) == {"type": "ready"}
    assert fake.connections == 1


async def test_busy_when_max_sessions_reached(fake):
    async with gateway(fake, max_sessions=1) as gw:
        async with connect(f"{gw.ws}/v1/transcribe") as first:
            assert await recv(first) == {"type": "ready"}
            async with connect(f"{gw.ws}/v1/transcribe") as second:
                events, close_code = await drain(second)
            assert events == [
                {
                    "type": "error",
                    "code": "busy",
                    "message": "too many active sessions, retry later",
                }
            ]
            assert close_code == 1013
        # The slot is released once the first session has closed its upstream socket.
        await wait_until(lambda: fake.active == 0)
        for _ in range(100):
            async with connect(f"{gw.ws}/v1/transcribe") as third:
                if (await recv(third))["type"] == "ready":
                    break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("the session slot was never released")


async def test_time_limit_still_sends_final(fake):
    async with (
        gateway(fake, max_session_seconds=0.4) as gw,
        connect(f"{gw.ws}/v1/transcribe") as ws,
    ):
        assert await recv(ws) == {"type": "ready"}
        for _ in range(3):
            await ws.send(PCM)
        events, close_code = await drain(ws)  # never sends stop
    assert [e["type"] for e in events] == ["delta", "final", "error"]
    assert events[1]["text"] == events[0]["text"]
    assert events[2]["code"] == "time_limit"
    assert close_code == 1000
    commits = [e for e in fake.received if e["type"] == "input_audio_buffer.commit"]
    assert commits[-1]["final"] is True


async def test_client_disconnect_without_stop_finishes_and_closes_upstream(fake):
    async with gateway(fake) as gw:
        async with connect(f"{gw.ws}/v1/transcribe") as ws:
            assert await recv(ws) == {"type": "ready"}
            await ws.send(PCM)
        # Client is gone: the gateway must send the final commit and drop the upstream socket.
        await wait_until(
            lambda: any(
                e["type"] == "input_audio_buffer.commit" and e.get("final") for e in fake.received
            )
        )
        await wait_until(lambda: fake.active == 0)


@pytest.mark.parametrize(
    "frame",
    [json.dumps({"type": "wat"}), "not json", b""],
    ids=["unknown-type", "not-json", "empty-audio"],
)
async def test_bad_request(fake, frame):
    async with gateway(fake) as gw, connect(f"{gw.ws}/v1/transcribe") as ws:
        assert await recv(ws) == {"type": "ready"}
        await ws.send(frame)
        events, close_code = await drain(ws)
    assert len(events) == 1 and events[0]["type"] == "error"
    assert events[0]["code"] == "bad_request"
    assert close_code == 1003


async def test_upstream_error_event_is_mapped(fake):
    fake.error_after_deltas = 1
    async with gateway(fake) as gw, connect(f"{gw.ws}/v1/transcribe") as ws:
        assert await recv(ws) == {"type": "ready"}
        for _ in range(3):
            await ws.send(PCM)
        events, close_code = await drain(ws)
    assert [e["type"] for e in events] == ["delta", "error"]
    assert events[-1]["code"] == "upstream_error"
    assert "processing_error" not in json.dumps(events)  # vLLM's codes never leak
    assert close_code == 1011


async def test_upstream_drop_is_mapped(fake):
    fake.drop_after_deltas = 1
    async with gateway(fake) as gw, connect(f"{gw.ws}/v1/transcribe") as ws:
        assert await recv(ws) == {"type": "ready"}
        for _ in range(3):
            await ws.send(PCM)
        events, close_code = await drain(ws)
    assert events[-1] == {
        "type": "error",
        "code": "upstream_error",
        "message": "the transcription engine failed",
    }
    assert close_code == 1011


async def test_upstream_unreachable_is_mapped():
    settings = replace(
        Settings.from_env({}), upstream_url=f"ws://127.0.0.1:{free_port()}/v1/realtime"
    )
    async with serve_app(create_app(settings)) as gw, connect(f"{gw.ws}/v1/transcribe") as ws:
        events, close_code = await drain(ws)
    assert events[0]["code"] == "upstream_error"
    assert close_code == 1011


async def test_health_is_503_until_the_engine_answers():
    port = free_port()
    settings = replace(Settings.from_env({}), upstream_url=f"ws://127.0.0.1:{port}/v1/realtime")
    async with serve_app(create_app(settings)) as gw, httpx.AsyncClient(base_url=gw.http) as http:
        response = await http.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "starting"}

        async with serve_app(FakeVLLM().app, port=port):
            response = await http.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


async def test_demo_page_and_static_files_are_served(fake):
    async with gateway(fake) as gw, httpx.AsyncClient(base_url=gw.http) as http:
        page = await http.get("/")
        assert page.status_code == 200
        assert "/static/mic.js" in page.text
        for name in ("mic.js", "pcm-worklet.js"):
            assert (await http.get(f"/static/{name}")).status_code == 200
