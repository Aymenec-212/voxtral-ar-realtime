"""Engine: owns the `vllm serve` subprocess. Callers never see its flags."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from voxtral_rt.config import Settings

log = logging.getLogger("voxtral_rt.engine")

HEALTH_PROBE_TIMEOUT_S = 2.0


class EngineError(Exception):
    pass


class Engine:
    def __init__(
        self,
        settings: Settings,
        model_dir: Path,
        executable: Sequence[str] = ("vllm", "serve"),
        stop_grace_s: float = 20.0,
    ) -> None:
        self._settings = settings
        self._model_dir = model_dir
        self._executable = tuple(executable)  # tests swap in a dummy command
        self._stop_grace_s = stop_grace_s
        self._process: subprocess.Popen | None = None

    @property
    def argv(self) -> list[str]:
        s = self._settings
        return [
            *self._executable,
            str(self._model_dir),
            # Bound to loopback only: the gateway is the sole public surface.
            "--host",
            "127.0.0.1",
            "--port",
            str(s.engine_port),
            # The gateway identifies the model by this exact name; vLLM validates it.
            "--served-model-name",
            s.model_id,
            # Required for Voxtral: its tokenizer only loads through mistral_common
            # (vLLM recipe for Voxtral-Mini-4B-Realtime-2602).
            "--tokenizer-mode",
            "mistral",
            # The model's default context is 131072 tokens; next to ~9 GB of weights on a 24 GB
            # L4 the KV cache cannot hold one such sequence and vLLM refuses to start. 16384
            # tokens (~80 ms of audio each) covers MAX_SESSION_SECONDS=300 with a wide margin.
            "--max-model-len",
            "16384",
            # Last, so VLLM_EXTRA_ARGS can override any default above.
            *shlex.split(s.vllm_extra_args),
        ]

    def start(self) -> None:
        log.info("starting engine: %s", shlex.join(self.argv))
        self._process = subprocess.Popen(self.argv)

    def wait_until_ready(self, timeout_s: float, poll_s: float = 1.0) -> None:
        """Block until vLLM answers /health; raise if it exits or the timeout passes."""
        if self._process is None:
            raise EngineError("engine not started")
        deadline = time.monotonic() + timeout_s
        while True:
            code = self._process.poll()
            if code is not None:
                raise EngineError(f"vllm exited with code {code} before becoming ready")
            if health_ok(self._settings.engine_health_url):
                log.info("engine is ready")
                return
            if time.monotonic() >= deadline:
                self.stop()
                raise EngineError(f"vllm not ready after {timeout_s:.0f}s")
            time.sleep(poll_s)

    def stop(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self._stop_grace_s)
        except subprocess.TimeoutExpired:
            log.warning("engine ignored SIGTERM for %.0fs, killing it", self._stop_grace_s)
            process.kill()
            process.wait()


def health_ok(url: str) -> bool:
    """True when an HTTP GET on `url` returns 2xx within a short timeout."""
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT_S) as response:
            return 200 <= response.status < 300
    except (OSError, ValueError):
        return False
