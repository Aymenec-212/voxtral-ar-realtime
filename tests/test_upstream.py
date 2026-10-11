import base64

import pytest

from tests.fake_vllm import DEFAULT_MODEL
from voxtral_rt.upstream import Delta, Final, UpstreamError, UpstreamSession

PCM = bytes(range(256)) * 12  # 3,072 bytes of arbitrary "audio"


async def collect(session: UpstreamSession) -> list[Delta | Final]:
    return [event async for event in session.events()]


async def test_handshake_order_then_deltas_then_final(fake):
    async with UpstreamSession.open(fake.url, DEFAULT_MODEL) as session:
        for _ in range(6):
            await session.send_audio(PCM)
        await session.finish()
        events = await collect(session)

    assert [type(e) for e in events] == [Delta, Delta, Final]
    assert events[-1].text == "".join(e.text for e in events[:-1])
    assert (events[-1].prompt_tokens, events[-1].completion_tokens) == (6, 2)

    kinds = [e["type"] for e in fake.received]
    assert kinds[:2] == ["session.update", "input_audio_buffer.commit"]
    assert fake.received[0]["model"] == DEFAULT_MODEL
    assert fake.received[1].get("final", False) is False
    assert kinds[-1] == "input_audio_buffer.commit" and fake.received[-1]["final"] is True


async def test_audio_is_sent_as_base64_pcm(fake):
    async with UpstreamSession.open(fake.url, DEFAULT_MODEL) as session:
        await session.send_audio(PCM)
        await session.finish()
        await collect(session)
    appends = [e for e in fake.received if e["type"] == "input_audio_buffer.append"]
    assert len(appends) == 1
    assert base64.b64decode(appends[0]["audio"]) == PCM


async def test_empty_audio_is_skipped(fake):
    async with UpstreamSession.open(fake.url, DEFAULT_MODEL) as session:
        await session.send_audio(b"")
        await session.finish()
        events = await collect(session)
    assert events == [Final("", prompt_tokens=0, completion_tokens=0)]
    assert not [e for e in fake.received if e["type"] == "input_audio_buffer.append"]


async def test_finish_is_idempotent(fake):
    async with UpstreamSession.open(fake.url, DEFAULT_MODEL) as session:
        await session.send_audio(PCM)
        await session.finish()
        await session.finish()
        events = await collect(session)
    assert events == [Final("", prompt_tokens=1, completion_tokens=0)]
    commits = [e for e in fake.received if e["type"] == "input_audio_buffer.commit"]
    finals = [e for e in commits if e.get("final")]
    assert len(finals) == 1


async def test_error_event_raises_upstream_error(fake):
    fake.error_after_deltas = 1
    async with UpstreamSession.open(fake.url, DEFAULT_MODEL) as session:
        for _ in range(3):
            await session.send_audio(PCM)
        with pytest.raises(UpstreamError) as info:
            await collect(session)
    assert info.value.code == "processing_error"


async def test_dropped_connection_raises_upstream_error(fake):
    fake.drop_after_deltas = 1
    async with UpstreamSession.open(fake.url, DEFAULT_MODEL) as session:
        for _ in range(3):
            await session.send_audio(PCM)
        with pytest.raises(UpstreamError) as info:
            await collect(session)
        await session.finish()  # must not raise on a dead socket
    assert info.value.code == "connection_closed"


async def test_wrong_model_raises_upstream_error(fake):
    async with UpstreamSession.open(fake.url, "not-a-model") as session:
        with pytest.raises(UpstreamError) as info:
            await collect(session)
    assert info.value.code == "model_not_found"


async def test_connect_failure_raises_upstream_error():
    from tests.conftest import free_port

    with pytest.raises(UpstreamError) as info:
        async with UpstreamSession.open(f"ws://127.0.0.1:{free_port()}/v1/realtime", DEFAULT_MODEL):
            pass
    assert info.value.code == "connect_failed"
