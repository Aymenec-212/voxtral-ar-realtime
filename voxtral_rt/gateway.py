"""The public surface: demo page, /health, /v1/metrics/stream and the /v1/transcribe WebSocket.

Client protocol (one utterance per connection): the server sends {"type": "ready"}; the client
sends binary PCM16 LE 16 kHz mono frames and finally the text frame {"type": "stop"}; the
server sends {"type": "delta", "text"} events, about once a second a {"type": "metrics", ...}
snapshot of the session (see metrics.SessionMetrics), then {"type": "final", "text", "metrics"}
and closes with 1000. Failures are {"type": "error", "code", "message"} followed by a close; the
codes are `busy` (1013), `time_limit` (1000, after a final), `bad_request` (1003),
`upstream_error` (1011).

GET /v1/metrics/stream is Server-Sent Events: one self-contained JSON snapshot of the whole
server (metrics.ServerMetrics plus vLLM's own counters) about once a second.

When API_KEY is set, both /v1/transcribe and /v1/metrics/stream require `?token=<key>`
(browsers cannot set headers on a WebSocket); a missing or wrong token closes the WebSocket
with 1008 and answers the stream with 401. GET / and GET /health stay public.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from voxtral_rt.config import Settings
from voxtral_rt.engine import health_ok
from voxtral_rt.metrics import EngineScraper, ServerMetrics, SessionMetrics
from voxtral_rt.upstream import Final, UpstreamError, UpstreamSession

log = logging.getLogger("voxtral_rt.gateway")

WEB_DIR = Path(__file__).parent / "web"
# After the time limit, how long we wait for the engine to flush the final transcript.
FINAL_GRACE_S = 10.0
# How often a session's metrics snapshot goes to its client and the SSE stream ticks.
METRICS_INTERVAL_S = 1.0

Outcome = Literal["final", "time_limit"]


class GateFull(Exception):
    pass


class SessionGate:
    """Counts active sessions and refuses new ones past the limit."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._active >= self._limit:
            raise GateFull
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1


def create_app(
    settings: Settings,
    open_upstream: Callable = UpstreamSession.open,
    engine_boot_s: float | None = None,
) -> FastAPI:
    app = FastAPI(title="voxtral-ar-realtime", docs_url=None, redoc_url=None)
    gate = SessionGate(settings.max_sessions)
    server = ServerMetrics(settings.max_sessions, engine_boot_s)
    scraper = EngineScraper(settings.engine_metrics_url)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/health")
    async def health() -> JSONResponse:
        ok = await asyncio.to_thread(health_ok, settings.engine_health_url)
        return JSONResponse({"status": "ok" if ok else "starting"}, status_code=200 if ok else 503)

    @app.get("/v1/metrics/stream")
    async def metrics_stream(request: Request, token: str | None = None) -> StreamingResponse:
        if not _authorized(settings.api_key, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        async def events() -> AsyncIterator[str]:
            while not await request.is_disconnected():
                engine = await asyncio.to_thread(scraper.scrape)
                snapshot = server.snapshot(active_sessions=gate.active, engine=engine)
                yield f"data: {json.dumps(snapshot)}\n\n"
                await asyncio.sleep(METRICS_INTERVAL_S)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.websocket("/v1/transcribe")
    async def transcribe(ws: WebSocket, token: str | None = None) -> None:
        await ws.accept()
        if not _authorized(settings.api_key, token):
            await ws.close(code=1008)
            return
        try:
            async with gate.slot():
                await _serve_session(ws, settings, open_upstream, server)
        except GateFull:
            server.record(None, "busy")
            await _fail(ws, "busy", "too many active sessions, retry later", 1013)

    return app


def _authorized(api_key: str | None, token: str | None) -> bool:
    """Check the `?token=` query parameter; no API_KEY configured means the API is open."""
    if not api_key:
        return True
    if token is None:
        return False
    # Constant time, on bytes so a non-ASCII token is a mismatch rather than a TypeError.
    return secrets.compare_digest(token.encode(), api_key.encode())


class _ClientGone(Exception):
    pass


class _BadRequest(Exception):
    pass


async def _serve_session(
    ws: WebSocket, settings: Settings, open_upstream: Callable, server: ServerMetrics
) -> None:
    metrics = SessionMetrics()
    outcome = "final"
    try:
        async with open_upstream(settings.engine_url, settings.model_id) as upstream:
            await _send(ws, {"type": "ready"})
            outcome = await _transcribe(ws, upstream, settings.max_session_seconds, metrics)
        if outcome == "time_limit":
            await _fail(ws, "time_limit", "session reached MAX_SESSION_SECONDS", 1000)
        else:
            await ws.close(code=1000)
    except _ClientGone:
        outcome = "client_gone"
    except _BadRequest as exc:
        outcome = "bad_request"
        await _fail(ws, "bad_request", str(exc), 1003)
    except UpstreamError as exc:
        outcome = "upstream_error"
        # The only place vLLM's own codes are visible: the log.
        log.warning("upstream failure: %s", exc)
        await _fail(ws, "upstream_error", "the transcription engine failed", 1011)
    finally:
        summary = metrics.snapshot()
        server.record(summary, outcome)
        log.info(json.dumps({"event": "session", "outcome": outcome, **summary}))


async def _transcribe(
    ws: WebSocket, upstream: UpstreamSession, max_seconds: float, metrics: SessionMetrics
) -> Outcome:
    """Run the pumps until the final transcript is delivered or the time limit hits."""
    pump = asyncio.create_task(_pump_client(ws, upstream, metrics))
    relay = asyncio.create_task(_relay_events(upstream, ws, metrics))
    ticker = asyncio.create_task(_tick_metrics(ws, metrics))
    try:
        try:
            async with asyncio.timeout(max_seconds):
                done, _ = await asyncio.wait({pump, relay}, return_when=asyncio.FIRST_COMPLETED)
                if pump in done:
                    pump.result()  # only ever ends by raising
                relay.result()
                return "final"
        except TimeoutError:
            metrics.audio_over()
            await upstream.finish()
            await asyncio.wait({relay}, timeout=FINAL_GRACE_S)
            return "time_limit"
    finally:
        for task in (pump, relay, ticker):
            task.cancel()
        await asyncio.gather(pump, relay, ticker, return_exceptions=True)


async def _pump_client(ws: WebSocket, upstream: UpstreamSession, metrics: SessionMetrics) -> None:
    """Client → upstream. Ends only by raising _ClientGone or _BadRequest."""
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            metrics.audio_over()
            await upstream.finish()
            raise _ClientGone
        pcm = message.get("bytes")
        if pcm is not None:
            if not pcm:
                raise _BadRequest("empty audio frame")
            metrics.audio_received(len(pcm))
            await upstream.send_audio(pcm)
            continue
        try:
            event = json.loads(message.get("text") or "")
        except json.JSONDecodeError:
            raise _BadRequest("text frames must be JSON") from None
        if isinstance(event, dict) and event.get("type") == "stop":
            metrics.audio_over()
            await upstream.finish()  # keep reading so a disconnect is still noticed
        else:
            raise _BadRequest('unknown text frame; send {"type": "stop"} to end the audio')


async def _relay_events(upstream: UpstreamSession, ws: WebSocket, metrics: SessionMetrics) -> None:
    """Upstream → client. Returns once the final transcript has been sent."""
    async for event in upstream.events():
        if isinstance(event, Final):
            metrics.final_sent(event.prompt_tokens, event.completion_tokens)
            await _send(ws, {"type": "final", "text": event.text, "metrics": metrics.snapshot()})
            return
        metrics.token(event.text)
        await _send(ws, {"type": "delta", "text": event.text})


async def _tick_metrics(ws: WebSocket, metrics: SessionMetrics) -> None:
    while True:
        await asyncio.sleep(METRICS_INTERVAL_S)
        await _send(ws, {"type": "metrics", **metrics.snapshot()})


async def _send(ws: WebSocket, payload: dict) -> None:
    connected = WebSocketState.CONNECTED
    if ws.client_state == connected and ws.application_state == connected:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def _fail(ws: WebSocket, code: str, message: str, close_code: int) -> None:
    with suppress(Exception):  # the client may already be gone; nothing more to report
        await _send(ws, {"type": "error", "code": code, "message": message})
        await ws.close(code=close_code)
