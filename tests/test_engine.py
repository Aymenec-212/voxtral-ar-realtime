import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tests.conftest import free_port
from voxtral_rt.config import Settings
from voxtral_rt.engine import Engine, EngineError

SLEEPER = (sys.executable, "-c", "import time; time.sleep(60)")
EXITS_3 = (sys.executable, "-c", "raise SystemExit(3)")
IGNORES_TERM = (
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
)


def test_argv_contains_model_dir_name_host_and_extra_args():
    extra = "--enforce-eager --max-model-len 4096"
    settings = replace(Settings.from_env({}), vllm_extra_args=extra)
    argv = Engine(settings, Path("/models/x")).argv
    assert argv[:2] == ["vllm", "serve"]
    assert "/models/x" in argv
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8001"
    assert argv[argv.index("--served-model-name") + 1] == settings.model_id
    assert argv[argv.index("--tokenizer-mode") + 1] == "mistral"
    assert argv[argv.index("--max-model-len") + 1] == "16384"
    # Extra args come last so they can override the defaults (vLLM takes the last value).
    assert argv[-3:] == ["--enforce-eager", "--max-model-len", "4096"]
    assert argv.index("--max-model-len") < argv.index("--enforce-eager")


async def test_wait_until_ready_succeeds_against_a_live_health_endpoint(fake, tmp_path):
    settings = replace(Settings.from_env({}), upstream_url=fake.url)
    engine = Engine(settings, tmp_path, executable=SLEEPER)
    engine.start()
    try:
        await asyncio.to_thread(engine.wait_until_ready, 5.0, 0.05)
    finally:
        engine.stop()


def test_wait_until_ready_fails_fast_when_the_process_exits(tmp_path):
    settings = replace(Settings.from_env({}), engine_port=free_port())
    engine = Engine(settings, tmp_path, executable=EXITS_3)
    engine.start()
    started = time.monotonic()
    with pytest.raises(EngineError, match="exited with code 3"):
        engine.wait_until_ready(30.0, poll_s=0.05)
    assert time.monotonic() - started < 5


def test_wait_until_ready_times_out_and_stops_the_process(tmp_path):
    settings = replace(Settings.from_env({}), engine_port=free_port())
    engine = Engine(settings, tmp_path, executable=SLEEPER)
    engine.start()
    with pytest.raises(EngineError, match="not ready"):
        engine.wait_until_ready(0.3, poll_s=0.05)
    assert engine._process.poll() is not None


def test_stop_kills_a_process_that_ignores_sigterm(tmp_path):
    engine = Engine(Settings.from_env({}), tmp_path, executable=IGNORES_TERM, stop_grace_s=0.5)
    engine.start()
    time.sleep(0.3)  # let the handler install
    started = time.monotonic()
    engine.stop()
    assert engine._process.poll() is not None
    assert time.monotonic() - started < 5
    engine.stop()  # idempotent
