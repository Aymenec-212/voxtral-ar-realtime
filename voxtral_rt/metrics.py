"""In-memory inference metrics: one session, the whole server, and vLLM's own counters.

Everything here lives in process memory and resets when the container restarts.
"""

from __future__ import annotations

import time
import urllib.request
from collections import Counter, deque
from collections.abc import Callable

# One output token per streaming step of 1000 / frame_rate ms of audio: VoxtralRealtimeBuffer
# yields `num_tokens = step / raw_audio_length_per_tok` per step (voxtral_realtime.py, vLLM
# v0.31.0); the model's tekken.json sets frame_rate 12.5, i.e. 80 ms, and checkpoint A measured
# exactly 12.5 tokens/s while audio streamed in real time.
SECONDS_PER_TOKEN = 0.08
# The model emits its first token only after transcription_delay_ms (480 ms = 6 frames) of
# audio, so lag and first-text figures include that built-in delay; it is not gateway overhead.
BUILTIN_DELAY_MS = 480
# PCM16 LE, mono, 16 kHz.
BYTES_PER_AUDIO_SECOND = 32_000
RATE_WINDOW_S = 5.0

# vLLM's Prometheus names (vllm/v1/metrics/loggers.py, v0.31.0). prometheus_client appends
# `_total` to counters in the text exposition, hence the suffix on generation_tokens.
KV_CACHE_USAGE = "vllm:kv_cache_usage_perc"  # gauge, 1.0 == 100 %
REQUESTS_RUNNING = "vllm:num_requests_running"
REQUESTS_WAITING = "vllm:num_requests_waiting"
GENERATION_TOKENS = "vllm:generation_tokens_total"  # counter

Clock = Callable[[], float]


class SessionMetrics:
    """Everything we can measure about one /v1/transcribe connection, in the gateway's clock.

    `token()` is called once per upstream delta event, empty text included: vLLM sends one
    delta per RequestOutput (connection.py, v0.31.0), and each output normally carries one
    token. When the producer outruns the consumer vLLM merges outputs, so this count is a lower
    bound; `completion_tokens` from the final usage is exact.
    """

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._first_audio_at: float | None = None
        self._audio_bytes = 0
        self._tokens = 0
        self._token_times: deque[float] = deque()
        self._first_text_at: float | None = None
        self._audio_over_at: float | None = None
        self._final_at: float | None = None
        self._max_lag_ms = 0.0
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    def audio_received(self, nbytes: int) -> None:
        if self._first_audio_at is None:
            self._first_audio_at = self._clock()
        self._audio_bytes += nbytes
        self._track_lag()

    def token(self, text: str) -> None:
        now = self._clock()
        self._tokens += 1
        self._token_times.append(now)
        if self._first_text_at is None and text.strip():
            self._first_text_at = now
        self._track_lag()

    def audio_over(self) -> None:
        """We just told the engine the audio is finished (stop, disconnect or time limit)."""
        if self._audio_over_at is None:
            self._audio_over_at = self._clock()

    def final_sent(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self._final_at = self._clock()
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def snapshot(self) -> dict:
        now = self._clock()
        received = self._audio_received_s()
        transcribed = min(self._tokens * SECONDS_PER_TOKEN, received)
        return {
            "session_s": round(now - self._started, 2),
            "audio_received_s": round(received, 2),
            "audio_transcribed_s": round(transcribed, 2),
            "tokens": self._tokens,
            "lag_ms": round((received - transcribed) * 1000, 1),
            "max_lag_ms": round(self._max_lag_ms, 1),
            "first_text_ms": self._ms_between(self._first_audio_at, self._first_text_at),
            "tokens_per_s": self._tokens_per_s(now),
            "finalize_ms": self._ms_between(self._audio_over_at, self._final_at),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def _audio_received_s(self) -> float:
        return self._audio_bytes / BYTES_PER_AUDIO_SECOND

    def _track_lag(self) -> None:
        received = self._audio_received_s()
        lag_ms = (received - min(self._tokens * SECONDS_PER_TOKEN, received)) * 1000
        self._max_lag_ms = max(self._max_lag_ms, lag_ms)

    def _tokens_per_s(self, now: float) -> float:
        if self._first_audio_at is None:
            return 0.0
        while self._token_times and self._token_times[0] < now - RATE_WINDOW_S:
            self._token_times.popleft()
        # The first second is too short to average over; clamp so the rate does not spike.
        window = max(1.0, min(RATE_WINDOW_S, now - self._first_audio_at))
        return round(len(self._token_times) / window, 2)

    @staticmethod
    def _ms_between(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000, 1)


ERROR_CODES = ("busy", "time_limit", "bad_request", "upstream_error")


class ServerMetrics:
    """Counters for the whole process plus percentiles over the last finished sessions."""

    def __init__(self, max_sessions: int, engine_boot_s: float | None, window: int = 100) -> None:
        self._max_sessions = max_sessions
        self._engine_boot_s = engine_boot_s
        self._recent: deque[dict] = deque(maxlen=window)
        self._completed = 0
        self._abandoned = 0
        self._errors: Counter[str] = Counter()

    def record(self, summary: dict | None, outcome: str) -> None:
        """`outcome` is "final", "client_gone" or one of the client-facing error codes."""
        if outcome in ERROR_CODES:
            self._errors[outcome] += 1
        if outcome in ("final", "time_limit"):
            self._completed += 1
        elif outcome == "client_gone":
            self._abandoned += 1
        if summary is not None:
            self._recent.append(summary)

    def snapshot(self, active_sessions: int, engine: dict) -> dict:
        return {
            "active_sessions": active_sessions,
            "max_sessions": self._max_sessions,
            "sessions_completed": self._completed,
            "sessions_abandoned": self._abandoned,
            "errors": {code: self._errors.get(code, 0) for code in ERROR_CODES},
            "recent_sessions": len(self._recent),
            "first_text_ms_p50": self._percentile("first_text_ms", 50),
            "first_text_ms_p95": self._percentile("first_text_ms", 95),
            "finalize_ms_p50": self._percentile("finalize_ms", 50),
            "finalize_ms_p95": self._percentile("finalize_ms", 95),
            "max_lag_ms_p95": self._percentile("max_lag_ms", 95),
            "engine": {"boot_s": self._engine_boot_s, **engine},
        }

    def _percentile(self, field: str, pct: int) -> float | None:
        values = sorted(s[field] for s in self._recent if s.get(field) is not None)
        if not values:
            return None
        # Nearest rank: fine for a window of 100 and avoids interpolation surprises.
        index = max(0, -(-pct * len(values) // 100) - 1)
        return values[index]


class EngineScraper:
    """Reads vLLM's Prometheus endpoint and reduces it to a few numbers.

    A metric vLLM does not expose (or an unreachable engine) yields None, never an error.
    """

    def __init__(
        self,
        metrics_url: str,
        fetch: Callable[[str], str] | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._url = metrics_url
        self._fetch = fetch or _fetch_text
        self._clock = clock
        self._last_tokens: tuple[float, float] | None = None  # (counter value, time)

    def scrape(self) -> dict:
        """Blocking; call it from a thread."""
        try:
            samples = _parse_prometheus(self._fetch(self._url))
        except (OSError, ValueError):
            return self._empty(reachable=False)
        now = self._clock()
        tokens = samples.get(GENERATION_TOKENS)
        rate = None
        if tokens is not None:
            if self._last_tokens is not None:
                previous, then = self._last_tokens
                if tokens >= previous and now > then:
                    rate = round((tokens - previous) / (now - then), 2)
            self._last_tokens = (tokens, now)
        usage = samples.get(KV_CACHE_USAGE)
        return {
            "reachable": True,
            "kv_cache_usage_pct": None if usage is None else round(usage * 100, 1),
            "requests_running": _as_int(samples.get(REQUESTS_RUNNING)),
            "requests_waiting": _as_int(samples.get(REQUESTS_WAITING)),
            "generation_tokens_per_s": rate,
        }

    @staticmethod
    def _empty(reachable: bool) -> dict:
        return {
            "reachable": reachable,
            "kv_cache_usage_pct": None,
            "requests_running": None,
            "requests_waiting": None,
            "generation_tokens_per_s": None,
        }


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return response.read().decode("utf-8", errors="replace")


def _parse_prometheus(text: str) -> dict[str, float]:
    """Sum every sample of each metric across its label sets (vLLM labels by model and engine)."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_end = len(line)
        for stop in ("{", " "):
            if (position := line.find(stop)) != -1:
                name_end = min(name_end, position)
        name = line[:name_end]
        try:
            value = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def _as_int(value: float | None) -> int | None:
    return None if value is None else int(value)
