from pathlib import Path

from voxtral_rt.config import Settings


def test_defaults_when_env_is_empty():
    s = Settings.from_env({})
    assert s.model_id == "mistralai/Voxtral-Mini-4B-Realtime-Arabic"
    assert s.models_dir == Path("/models")
    assert s.port == 8000
    assert s.upstream_url is None
    assert s.api_key is None
    assert s.engine_url == "ws://127.0.0.1:8001/v1/realtime"
    assert s.engine_health_url == "http://127.0.0.1:8001/health"


def test_env_overrides_and_upstream_url_disables_engine():
    s = Settings.from_env(
        {
            "PORT": "9100",
            "UPSTREAM_URL": "ws://localhost:9000/v1/realtime",
            "API_KEY": "secret",
            "MAX_SESSIONS": "2",
            "MAX_SESSION_SECONDS": "12.5",
            "VLLM_EXTRA_ARGS": "--enforce-eager",
        }
    )
    assert s.port == 9100
    assert s.engine_url == "ws://localhost:9000/v1/realtime"
    assert s.engine_health_url == "http://localhost:9000/health"
    assert s.api_key == "secret"
    assert s.max_sessions == 2
    assert s.max_session_seconds == 12.5
    assert s.vllm_extra_args == "--enforce-eager"


def test_blank_values_fall_back_to_defaults():
    s = Settings.from_env({"PORT": "  ", "API_KEY": ""})
    assert s.port == 8000
    assert s.api_key is None
