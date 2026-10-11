"""`python -m voxtral_rt`: weights → engine → gateway."""

import logging

import uvicorn

from voxtral_rt.config import Settings
from voxtral_rt.engine import Engine
from voxtral_rt.gateway import create_app
from voxtral_rt.weights import ensure_weights


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = Settings.from_env()
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
        uvicorn.run(create_app(settings), host="0.0.0.0", port=settings.port, log_level="info")
    finally:
        if engine is not None:
            engine.stop()


if __name__ == "__main__":
    main()
