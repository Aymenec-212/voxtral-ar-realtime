"""Modal deployment: the reference place where the real model runs.

    modal run deploy/modal_app.py::download_weights   # once, CPU only, fills the volume
    modal serve deploy/modal_app.py                    # dev URL, stops on Ctrl+C
    modal deploy deploy/modal_app.py                   # persistent URL, scales to zero

Pinned to modal==1.6.1 (see pyproject.toml). Set VOXTRAL_IMAGE to a published image reference
(ghcr.io/<owner>/voxtral-ar-realtime:<tag>) to run it instead of building docker/Dockerfile.
"""

import os
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = "/models"
PORT = 8000
MINUTES = 60

app = modal.App("voxtral-ar-realtime")
volume = modal.Volume.from_name("voxtral-models", create_if_missing=True)

if os.environ.get("VOXTRAL_IMAGE"):
    image = modal.Image.from_registry(os.environ["VOXTRAL_IMAGE"]).entrypoint([])
else:
    image = modal.Image.from_dockerfile(
        REPO_ROOT / "docker" / "Dockerfile", context_dir=REPO_ROOT
    ).entrypoint([])

# Everything voxtral_rt reads from the environment. Add API_KEY / VLLM_EXTRA_ARGS here.
ENV = {
    "MODELS_DIR": MODELS_DIR,
    "PORT": str(PORT),
}


@app.function(image=image, volumes={MODELS_DIR: volume}, timeout=60 * MINUTES, env=ENV)
def download_weights() -> None:
    """CPU-only: fill the volume once so GPU time is never spent downloading."""
    from voxtral_rt.config import Settings
    from voxtral_rt.weights import ensure_weights

    settings = Settings.from_env()
    path = ensure_weights(settings.model_id, settings.model_revision, Path(MODELS_DIR))
    volume.commit()
    total = 0
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        size = file.stat().st_size
        total += size
        print(f"{size / 2**30:8.2f} GiB  {file.relative_to(path)}")
    print(f"{total / 2**30:8.2f} GiB  total in {path}")


@app.server(
    image=image,
    gpu="L4",
    volumes={MODELS_DIR: volume},
    env=ENV,
    port=PORT,
    max_containers=1,  # hard cost cap
    scaledown_window=5 * MINUTES,
    # voxtral_rt gives up after ENGINE_READY_TIMEOUT_S (900 s) with a clear error; Modal's own
    # limit is set later so that message, not a Modal timeout, is what we see.
    startup_timeout=20 * MINUTES,
    target_concurrency=16,  # the page plus several WebSockets share the one container
    unauthenticated=True,  # the gateway's own API_KEY is the auth layer, if set
)
class Server:
    # Assumption (Modal docs unreachable from the dev sandbox): Modal 1.6's class-based
    # @app.server is the current replacement for @modal.web_server and proxies WebSockets.
    # Checkpoint A verifies this; the fallback is @app.function + @modal.web_server(PORT).
    @modal.enter()
    def start(self) -> None:
        self.process = subprocess.Popen(["python3", "-m", "voxtral_rt"])

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
