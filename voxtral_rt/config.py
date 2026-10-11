"""Settings: every knob the service has, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-Arabic"
# Pinned commit of the Hugging Face repo, current as of 9 Oct 2026.
DEFAULT_MODEL_REVISION = "a8b0156f0dd61e2fcfe73e16dbbee600c4a967f1"


@dataclass(frozen=True)
class Settings:
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    models_dir: Path = Path("/models")
    port: int = 8000
    engine_port: int = 8001
    # When set, the launcher skips weights and engine and the gateway talks to this URL.
    upstream_url: str | None = None
    api_key: str | None = None
    max_sessions: int = 4
    max_session_seconds: float = 300.0
    engine_ready_timeout_s: float = 900.0
    vllm_extra_args: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        def get(name: str, default: str) -> str:
            value = env.get(name, "").strip()
            return value or default

        return cls(
            model_id=get("MODEL_ID", DEFAULT_MODEL_ID),
            model_revision=get("MODEL_REVISION", DEFAULT_MODEL_REVISION),
            models_dir=Path(get("MODELS_DIR", "/models")),
            port=int(get("PORT", "8000")),
            engine_port=int(get("ENGINE_PORT", "8001")),
            upstream_url=env.get("UPSTREAM_URL", "").strip() or None,
            api_key=env.get("API_KEY", "").strip() or None,
            max_sessions=int(get("MAX_SESSIONS", "4")),
            max_session_seconds=float(get("MAX_SESSION_SECONDS", "300")),
            engine_ready_timeout_s=float(get("ENGINE_READY_TIMEOUT_S", "900")),
            vllm_extra_args=env.get("VLLM_EXTRA_ARGS", ""),
        )

    @property
    def engine_url(self) -> str:
        """The realtime WebSocket URL the gateway talks to."""
        return self.upstream_url or f"ws://127.0.0.1:{self.engine_port}/v1/realtime"

    @property
    def engine_health_url(self) -> str:
        """The engine's HTTP health URL, derived from the realtime URL (same host and port)."""
        parts = urlsplit(self.engine_url)
        scheme = "https" if parts.scheme == "wss" else "http"
        return urlunsplit((scheme, parts.netloc, "/health", "", ""))
