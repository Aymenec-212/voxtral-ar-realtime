"""The vLLM protocol must stay inside upstream.py (and the fake that mirrors it)."""

from pathlib import Path

VLLM_EVENT_NAMES = (
    "session.created",
    "session.update",
    "input_audio_buffer",
    "transcription.delta",
    "transcription.done",
)


def test_only_upstream_knows_vllm_event_names():
    package = Path(__file__).resolve().parents[1] / "voxtral_rt"
    offenders = [
        f"{path.name}: {name}"
        for path in package.rglob("*.py")
        if path.name != "upstream.py"
        for name in VLLM_EVENT_NAMES
        if name in path.read_text()
    ]
    assert offenders == []
