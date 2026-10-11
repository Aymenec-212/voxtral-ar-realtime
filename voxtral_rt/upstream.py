"""UpstreamSession: the only module that knows vLLM's realtime WebSocket protocol.

Protocol (vLLM v0.31.0, vllm/entrypoints/speech_to_text/realtime/): JSON text frames; the
server sends `session.created`, the client sends `session.update` with the model name, then a
non-final `input_audio_buffer.commit` to start generation, then base64 PCM16 chunks as
`input_audio_buffer.append`; `transcription.delta` events stream back; a final commit ends the
utterance with `transcription.done`. Errors arrive as `{"type": "error", "error", "code"}`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Final:
    text: str


class UpstreamError(Exception):
    """vLLM reported an error or the connection to it was lost."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class UpstreamSession:
    """One utterance on one vLLM realtime connection."""

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        self._finished = False

    @classmethod
    @asynccontextmanager
    async def open(cls, url: str, model: str) -> AsyncIterator[UpstreamSession]:
        """Connect, wait for session.created, send session.update + the starting commit."""
        try:
            # max_size=None: a transcript is tiny, but never let a library default close us.
            ws = await connect(url, max_size=None)
        except (OSError, WebSocketException) as exc:
            raise UpstreamError("connect_failed", str(exc)) from exc
        try:
            session = cls(ws)
            await session._handshake(model)
            yield session
        finally:
            await ws.close()

    async def send_audio(self, pcm: bytes) -> None:
        """Forward one chunk of raw PCM16 LE 16 kHz mono audio. Empty chunks are skipped."""
        if not pcm or self._finished:
            return
        audio = base64.b64encode(pcm).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": audio})

    async def finish(self) -> None:
        """Tell vLLM the audio is over so it emits the final transcript. Idempotent."""
        if self._finished:
            return
        self._finished = True
        # A dead socket is reported by events(); there is nothing left to finish here.
        with suppress(ConnectionClosed):
            await self._send({"type": "input_audio_buffer.commit", "final": True})

    async def events(self) -> AsyncIterator[Delta | Final]:
        """Yield Delta events and then exactly one Final. Raises UpstreamError on failure."""
        try:
            async for raw in self._ws:
                event = _parse(raw)
                kind = event.get("type")
                if kind == "transcription.delta":
                    yield Delta(event.get("delta", ""))
                elif kind == "transcription.done":
                    yield Final(event.get("text", ""))
                    return
                elif kind == "error":
                    raise UpstreamError(event.get("code") or "unknown", event.get("error", ""))
                # Any other event (e.g. a second session.created) is irrelevant to us.
        except ConnectionClosed as exc:
            raise UpstreamError("connection_closed", str(exc)) from exc
        raise UpstreamError("connection_closed", "upstream ended before transcription.done")

    async def _handshake(self, model: str) -> None:
        try:
            first = _parse(await self._ws.recv())
        except ConnectionClosed as exc:
            raise UpstreamError("connection_closed", str(exc)) from exc
        if first.get("type") == "error":
            raise UpstreamError(first.get("code") or "unknown", first.get("error", ""))
        if first.get("type") != "session.created":
            raise UpstreamError("protocol", f"expected session.created, got {first.get('type')!r}")
        # vLLM validates the name silently; a mismatch surfaces as an error in events().
        await self._send({"type": "session.update", "model": model})
        await self._send({"type": "input_audio_buffer.commit"})

    async def _send(self, event: dict) -> None:
        await self._ws.send(json.dumps(event))


def _parse(raw: str | bytes) -> dict:
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UpstreamError("protocol", "upstream sent a non-JSON frame") from exc
    return event if isinstance(event, dict) else {}
