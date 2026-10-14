"""`python -m voxtral_rt`: weights → engine → gateway."""

import logging
import re

import uvicorn

from voxtral_rt.config import Settings
from voxtral_rt.engine import Engine
from voxtral_rt.gateway import create_app
from voxtral_rt.weights import ensure_weights

log = logging.getLogger("voxtral_rt")

_TOKEN = re.compile(r"(token=)[^&\s\"]+")


class RedactToken(logging.Filter):
    """Rewrite `token=<value>` to `token=***` so the API key never reaches the logs.

    uvicorn logs request paths with their query string: HTTP requests on "uvicorn.access" and
    the WebSocket `"WebSocket /v1/transcribe?token=…" [accepted]` line on "uvicorn.error". The
    path is a %-argument, so the arguments are redacted too; uvicorn's access formatter unpacks
    them, which is why they stay a tuple of the same shape.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _TOKEN.sub(r"\1***", record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                _TOKEN.sub(r"\1***", arg) if isinstance(arg, str) else arg for arg in record.args
            )
        return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(RedactToken())
    settings = Settings.from_env()
    if not settings.api_key:
        log.warning(
            "API_KEY is not set: authentication is disabled, anyone who can reach the port can "
            "use the API. Set API_KEY for any deployment others can reach."
        )
    engine: Engine | None = None
    if settings.upstream_url is None:
        model_dir = ensure_weights(settings.model_id, settings.model_revision, settings.models_dir)
        engine = Engine(settings, model_dir)
        engine.start()
        try:
            engine.wait_until_ready(settings.engine_ready_timeout_s)
        except BaseException:
            engine.stop()
            raise
    try:
        app = create_app(settings, engine_boot_s=engine.boot_seconds if engine else None)
        uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")
    finally:
        if engine is not None:
            engine.stop()


if __name__ == "__main__":
    main()
