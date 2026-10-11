#!/usr/bin/env python3
"""Stream a 16 kHz mono PCM16 WAV file to the gateway at real-time pace and print the deltas.

    python clients/stream_wav.py --url ws://localhost:8000/v1/transcribe --wav sample.wav

Convert anything else first:  ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 sample.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

SAMPLE_RATE = 16_000


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            sys.exit(
                f"{path}: need 16 kHz mono 16-bit PCM, got {wav.getframerate()} Hz, "
                f"{wav.getnchannels()} ch, {8 * wav.getsampwidth()}-bit. Convert with:\n"
                f"  ffmpeg -i {path} -ac 1 -ar 16000 -sample_fmt s16 out.wav"
            )
        return wav.readframes(wav.getnframes())


async def send_audio(ws, pcm: bytes, chunk_ms: int, realtime: bool) -> None:
    chunk_bytes = SAMPLE_RATE * 2 * chunk_ms // 1000
    started = time.monotonic()
    for index, offset in enumerate(range(0, len(pcm), chunk_bytes)):
        await ws.send(pcm[offset : offset + chunk_bytes])
        if realtime:
            due = started + (index + 1) * chunk_ms / 1000
            await asyncio.sleep(max(0.0, due - time.monotonic()))
    await ws.send(json.dumps({"type": "stop"}))


async def run(url: str, pcm: bytes, chunk_ms: int, realtime: bool) -> int:
    started = time.monotonic()
    first_delta_at: float | None = None
    async with connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            print(f"server refused: {ready}", file=sys.stderr)
            return 1
        sender = asyncio.create_task(send_audio(ws, pcm, chunk_ms, realtime))
        try:
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "delta":
                    if first_delta_at is None:
                        first_delta_at = time.monotonic() - started
                        print(f"[first delta after {first_delta_at:.2f}s]", file=sys.stderr)
                    print(event["text"], end="", flush=True)
                elif event["type"] == "final":
                    print(f"\n[final] {event['text']}")
                elif event["type"] == "error":
                    print(f"\n[error] {event['code']}: {event['message']}", file=sys.stderr)
                    if event["code"] != "time_limit":
                        return 1
        except ConnectionClosed as exc:
            if exc.rcvd is None or exc.rcvd.code != 1000:
                print(f"\n[closed] {exc}", file=sys.stderr)
                return 1
        finally:
            sender.cancel()
    print(f"[done in {time.monotonic() - started:.2f}s]", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://localhost:8000/v1/transcribe")
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--token", help="API key, if the server requires one")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--fast", action="store_true", help="send as fast as possible")
    args = parser.parse_args(argv)
    url = args.url + (f"?token={args.token}" if args.token else "")
    pcm = read_pcm(args.wav)
    print(f"streaming {len(pcm) / (SAMPLE_RATE * 2):.1f}s of audio to {args.url}", file=sys.stderr)
    return asyncio.run(run(url, pcm, args.chunk_ms, realtime=not args.fast))


if __name__ == "__main__":
    sys.exit(main())
