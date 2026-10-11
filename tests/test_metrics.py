import pytest

from voxtral_rt.metrics import (
    GENERATION_TOKENS,
    KV_CACHE_USAGE,
    REQUESTS_RUNNING,
    REQUESTS_WAITING,
    EngineScraper,
    ServerMetrics,
    SessionMetrics,
)

CHUNK = 3200  # 100 ms of PCM16 mono 16 kHz


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_session_lag_first_text_and_finalize():
    clock = Clock()
    m = SessionMetrics(clock)
    assert m.snapshot()["tokens_per_s"] == 0.0

    # 2 s of audio, the model has produced nothing yet: lag is the whole 2 s.
    for _ in range(20):
        m.audio_received(CHUNK)
        clock.advance(0.1)
    snap = m.snapshot()
    assert snap["audio_received_s"] == 2.0
    assert snap["audio_transcribed_s"] == 0.0
    assert snap["lag_ms"] == 2000.0
    assert snap["first_text_ms"] is None

    # 19 tokens (1.52 s of audio) catch up most of it: lag = 2.0 - 1.52 = 0.48 s.
    m.token("")
    for _ in range(18):
        m.token("كلمة")
    snap = m.snapshot()
    assert snap["tokens"] == 19
    assert snap["audio_transcribed_s"] == 1.52
    assert snap["lag_ms"] == 480.0
    assert snap["max_lag_ms"] == 2000.0
    assert snap["first_text_ms"] == 2000.0  # the first non-empty delta, 2 s after t0

    # Transcribed audio can never exceed received audio.
    for _ in range(50):
        m.token("x")
    assert m.snapshot()["audio_transcribed_s"] == 2.0
    assert m.snapshot()["lag_ms"] == 0.0

    m.audio_over()
    clock.advance(0.25)
    m.final_sent(prompt_tokens=20, completion_tokens=69)
    snap = m.snapshot()
    assert snap["finalize_ms"] == 250.0
    assert snap["prompt_tokens"] == 20
    assert snap["completion_tokens"] == 69
    assert snap["session_s"] == pytest.approx(2.25)


def test_session_tokens_per_s_uses_a_five_second_window():
    clock = Clock()
    m = SessionMetrics(clock)
    m.audio_received(CHUNK)
    for _ in range(100):  # 12.5 tokens/s for 8 s
        m.token("x")
        clock.advance(0.08)
    assert m.snapshot()["tokens_per_s"] == pytest.approx(12.5, abs=0.3)
    clock.advance(10)  # the GPU stalls: the window empties
    assert m.snapshot()["tokens_per_s"] == 0.0


def test_server_counts_and_percentiles():
    server = ServerMetrics(max_sessions=4, engine_boot_s=285.0, window=100)
    for i in range(1, 11):
        server.record({"first_text_ms": i * 100.0, "finalize_ms": None, "max_lag_ms": i}, "final")
    server.record({"first_text_ms": None, "finalize_ms": 50.0, "max_lag_ms": 99}, "time_limit")
    server.record(None, "busy")
    server.record({"first_text_ms": None, "finalize_ms": None, "max_lag_ms": 0}, "client_gone")
    server.record({"first_text_ms": None, "finalize_ms": None, "max_lag_ms": 0}, "upstream_error")

    snap = server.snapshot(active_sessions=2, engine={"reachable": False})
    assert snap["active_sessions"] == 2
    assert snap["max_sessions"] == 4
    assert snap["sessions_completed"] == 11
    assert snap["sessions_abandoned"] == 1
    assert snap["errors"] == {"busy": 1, "time_limit": 1, "bad_request": 0, "upstream_error": 1}
    assert snap["first_text_ms_p50"] == 500.0
    assert snap["first_text_ms_p95"] == 1000.0
    assert snap["finalize_ms_p50"] == 50.0
    assert snap["max_lag_ms_p95"] == 99
    assert snap["engine"] == {"boot_s": 285.0, "reachable": False}


def test_server_percentiles_are_none_without_data():
    snap = ServerMetrics(4, None).snapshot(active_sessions=0, engine={})
    assert snap["first_text_ms_p50"] is None
    assert snap["engine"] == {"boot_s": None}


EXPOSITION = f"""
# HELP {KV_CACHE_USAGE} KV-cache usage. 1 means 100 percent usage.
# TYPE {KV_CACHE_USAGE} gauge
{KV_CACHE_USAGE}{{engine="0",model_name="m"}} 0.3489
{REQUESTS_RUNNING}{{engine="0",model_name="m"}} 2.0
{REQUESTS_WAITING}{{engine="0",model_name="m"}} 1.0
{GENERATION_TOKENS}{{engine="0",model_name="m"}} 1000.0
{GENERATION_TOKENS}{{engine="1",model_name="m"}} 500.0
vllm:generation_tokens_created{{engine="0",model_name="m"}} 1.79e+09
"""


def test_scraper_reads_the_pinned_metrics_and_rates_the_counter():
    clock = Clock()
    pages = iter([EXPOSITION, EXPOSITION.replace("1000.0", "1250.0")])
    scraper = EngineScraper("http://engine/metrics", fetch=lambda url: next(pages), clock=clock)

    first = scraper.scrape()
    assert first == {
        "reachable": True,
        "kv_cache_usage_pct": 34.9,
        "requests_running": 2,
        "requests_waiting": 1,
        "generation_tokens_per_s": None,  # needs two scrapes
    }
    clock.advance(20)
    assert scraper.scrape()["generation_tokens_per_s"] == 12.5  # (1750 - 1500) / 20


def test_scraper_tolerates_missing_metrics_and_an_unreachable_engine():
    partial = f'{REQUESTS_RUNNING}{{engine="0"}} 1.0\n'
    scraper = EngineScraper("http://engine/metrics", fetch=lambda url: partial)
    snap = scraper.scrape()
    assert snap["requests_running"] == 1
    assert snap["kv_cache_usage_pct"] is None
    assert snap["generation_tokens_per_s"] is None

    def down(url: str) -> str:
        raise OSError("connection refused")

    snap = EngineScraper("http://engine/metrics", fetch=down).scrape()
    assert snap["reachable"] is False
    assert all(v is None for k, v in snap.items() if k != "reachable")
