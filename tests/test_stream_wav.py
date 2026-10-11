import asyncio
import math
import struct
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path

from tests.conftest import serve_app
from voxtral_rt.config import Settings
from voxtral_rt.gateway import create_app

ROOT = Path(__file__).resolve().parents[1]


def write_tone(path: Path, seconds: float, rate: int = 16_000, channels: int = 1) -> None:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
        for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)


async def run_client(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ROOT / "clients" / "stream_wav.py"), *args]
    return await asyncio.to_thread(
        subprocess.run, cmd, capture_output=True, text=True, timeout=30, cwd=ROOT
    )


async def test_stream_wav_prints_deltas_and_final(fake, tmp_path):
    wav = tmp_path / "tone.wav"
    write_tone(wav, seconds=0.9)  # 9 chunks of 100 ms → 3 fake deltas
    settings = replace(Settings.from_env({}), upstream_url=fake.url)
    async with serve_app(create_app(settings)) as gw:
        result = await run_client("--url", f"{gw.ws}/v1/transcribe", "--wav", str(wav), "--fast")
    assert result.returncode == 0, result.stderr
    assert "[final] كلمة1 كلمة2 كلمة3" in result.stdout
    assert "[first delta after" in result.stderr


async def test_stream_wav_rejects_wrong_format(tmp_path):
    wav = tmp_path / "stereo.wav"
    write_tone(wav, seconds=0.2, rate=44_100, channels=2)
    result = await run_client("--url", "ws://127.0.0.1:1/v1/transcribe", "--wav", str(wav))
    assert result.returncode != 0
    assert "need 16 kHz mono 16-bit PCM" in result.stderr
    assert "ffmpeg" in result.stderr
