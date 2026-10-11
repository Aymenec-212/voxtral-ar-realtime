"""A CPU stand-in for vLLM's realtime WebSocket server.

It speaks the exact upstream protocol (vllm/entrypoints/speech_to_text/realtime/*.py at
v0.31.0): JSON text frames only, `session.created` on connect, `session.update` to validate
the model, a non-final `input_audio_buffer.commit` to start generation, base64 PCM16
`input_audio_buffer.append` chunks, `transcription.delta` events, and `transcription.done`
after a final commit. Two switches let tests inject an error event or drop the socket
mid-stream. Run standalone with `python -m tests.fake_vllm --port 9000`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import time

from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

DEFAULT_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-Arabic"


class FakeVLLM:
    """Holds the fake's configuration, its FastAPI app and a log of everything it received."""

    def __init__(self, model: str = DEFAULT_MODEL, chunks_per_delta: int = 3) -> None:
        self.model = model
        self.chunks_per_delta = chunks_per_delta
        # Test switches: after this many deltas, send an error event / close the socket.
        self.error_after_deltas: int | None = None
        self.drop_after_deltas: int | None = None
        # Like vLLM, send a delta for every "token" even when it carries no text.
        self.empty_deltas = False
        self.generated_tokens = 0  # feeds the fake /metrics counter
        self.generating = 0
        # Every client event, in arrival order, so tests can assert on protocol order.
        self.received: list[dict] = []
        self.connections = 0
        self.active = 0
        self.app = FastAPI()
        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/metrics", self._metrics, methods=["GET"])
        self.app.add_api_websocket_route("/v1/realtime", self._realtime)

    async def _health(self) -> dict:
        return {}

    async def _metrics(self) -> PlainTextResponse:
        """The subset of vLLM's Prometheus exposition that the gateway reads, same names."""
        labels = f'{{engine="0",model_name="{self.model}"}}'
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{labels} {self.generating}.0",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{labels} 0.0",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f"vllm:kv_cache_usage_perc{labels} {0.05 * self.generating}",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total{labels} {self.generated_tokens}.0",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    async def _realtime(self, websocket: WebSocket) -> None:
        self.connections += 1
        self.active += 1
        try:
            await _Connection(self, websocket).run()
        finally:
            self.active -= 1


class _Connection:
    def __init__(self, fake: FakeVLLM, ws: WebSocket) -> None:
        self.fake = fake
        self.ws = ws
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.generation: asyncio.Task | None = None
        self.validated = False

    async def run(self) -> None:
        await self.ws.accept()
        created = {"type": "session.created", "id": "sess-fake", "created": int(time.time())}
        await self._send(created)
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is None:
                    # vLLM calls receive_text(), which blows up on binary frames and ends
                    # the connection; we close explicitly with the same net effect.
                    await self.ws.close(code=1003)
                    break
                try:
                    event = json.loads(message["text"])
                except json.JSONDecodeError:
                    await self._error("Invalid JSON", "invalid_json")
                    continue
                self.fake.received.append(event)
                await self._handle(event)
        except WebSocketDisconnect:
            pass
        finally:
            self.queue.put_nowait(None)
            if self.generation and not self.generation.done():
                self.generation.cancel()

    async def _handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "session.update":
            model = event.get("model")
            if model is None:
                await self._error("Missing required field: model", "invalid_event")
            elif model != self.fake.model:
                await self._error(f"The model `{model}` does not exist.", "model_not_found")
            else:
                self.validated = True
        elif kind == "input_audio_buffer.append":
            try:
                audio = base64.b64decode(event.get("audio", ""), validate=True)
                if not audio:
                    raise ValueError("empty audio")
            except (binascii.Error, ValueError, TypeError):
                await self._error("Invalid audio data", "invalid_audio")
                return
            self.queue.put_nowait(audio)
        elif kind == "input_audio_buffer.commit":
            if not self.validated:
                await self._error("Model not validated.", "model_not_validated")
            elif event.get("final", False):
                self.queue.put_nowait(None)
            elif self.generation is None or self.generation.done():
                self.generation = asyncio.create_task(self._generate())
            # else: a commit while generating is ignored, as in vLLM.
        else:
            await self._error(f"Unknown event type: {kind}", "unknown_event")

    async def _generate(self) -> None:
        self.fake.generating += 1
        try:
            await self._generate_until_final()
        finally:
            self.fake.generating -= 1

    async def _generate_until_final(self) -> None:
        text = ""
        chunks = 0
        deltas = 0  # text-bearing deltas
        tokens = 0  # every delta sent, empty ones included
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                break
            chunks += 1
            if chunks % self.fake.chunks_per_delta:
                if self.fake.empty_deltas:
                    tokens += 1
                    self.fake.generated_tokens += 1
                    await self._send({"type": "transcription.delta", "delta": ""})
                continue
            deltas += 1
            tokens += 1
            self.fake.generated_tokens += 1
            delta = f"كلمة{deltas} "
            text += delta
            await self._send({"type": "transcription.delta", "delta": delta})
            if self.fake.error_after_deltas is not None and deltas >= self.fake.error_after_deltas:
                await self._error("injected failure", "processing_error")
                return
            if self.fake.drop_after_deltas is not None and deltas >= self.fake.drop_after_deltas:
                await self.ws.close(code=1011)
                return
        usage = {
            "prompt_tokens": chunks,
            "completion_tokens": tokens,
            "total_tokens": chunks + tokens,
        }
        await self._send({"type": "transcription.done", "text": text, "usage": usage})
        while not self.queue.empty():
            self.queue.get_nowait()

    async def _send(self, event: dict) -> None:
        if self.ws.application_state == WebSocketState.CONNECTED:
            await self.ws.send_text(json.dumps(event))

    async def _error(self, message: str, code: str) -> None:
        await self._send({"type": "error", "error": message, "code": code})


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Fake vLLM realtime server (CPU only)")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunks-per-delta", type=int, default=3)
    args = parser.parse_args()
    fake = FakeVLLM(model=args.model, chunks_per_delta=args.chunks_per_delta)
    uvicorn.run(fake.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
