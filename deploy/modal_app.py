"""Modal deployment: builds docker/Dockerfile and serves it on one L4 GPU.

    KEY=$(openssl rand -hex 16); modal secret create voxtral-api-key API_KEY="$KEY"   # once
    modal run deploy/modal_app.py::download_weights   # once, CPU only, fills the volume
    modal serve deploy/modal_app.py                    # dev URL, stops on Ctrl+C
    modal deploy deploy/modal_app.py                   # persistent URL, scales to zero

Pinned to modal==1.6.1 (see pyproject.toml). The endpoint always requires the API key: the
server refuses to start without the `voxtral-api-key` secret.
"""

import os
import socket
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = "/models"
PORT = 8000
MINUTES = 60
API_KEY_SECRET = "voxtral-api-key"
CREATE_SECRET_COMMAND = (
    f'KEY=$(openssl rand -hex 16); modal secret create {API_KEY_SECRET} API_KEY="$KEY"'
)

app = modal.App("voxtral-ar-realtime")
volume = modal.Volume.from_name("voxtral-models", create_if_missing=True)
image = modal.Image.from_dockerfile(
    REPO_ROOT / "docker" / "Dockerfile", context_dir=REPO_ROOT
).entrypoint([])

# Everything voxtral_rt reads from the environment, except API_KEY, which comes from the secret.
# Add VLLM_EXTRA_ARGS or MAX_SESSIONS here.
ENV = {
    "MODELS_DIR": MODELS_DIR,
    "PORT": str(PORT),
}


@app.function(
    image=image,
    volumes={MODELS_DIR: volume},
    timeout=60 * MINUTES,
    env=ENV,
    # The default fraction of a core starves the high-performance HF downloader.
    cpu=2,
    memory=4096,
)
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
    # Modal refuses to start the app if this secret or its API_KEY entry is missing; create it
    # with CREATE_SECRET_COMMAND. There is deliberately no way to run this endpoint keyless.
    secrets=[modal.Secret.from_name(API_KEY_SECRET, required_keys=["API_KEY"])],
    port=PORT,
    max_containers=1,  # hard cost cap
    scaledown_window=5 * MINUTES,
    # voxtral_rt gives up after ENGINE_READY_TIMEOUT_S (900 s) with a clear error; Modal's own
    # limit is set later so that message, not a Modal timeout, is what we see.
    startup_timeout=20 * MINUTES,
    target_concurrency=16,  # the page plus several WebSockets share the one container
    unauthenticated=True,  # Modal's proxy auth is off; the gateway's API_KEY is the auth layer
)
class Server:
    # @app.server proxies WebSockets: /v1/transcribe has been verified end to end on Modal.
    @modal.enter()
    def start(self) -> None:
        require_api_key(os.environ)
        self.process = subprocess.Popen(["python3", "-m", "voxtral_rt"])
        wait_until_listening(self.process, PORT)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()


def require_api_key(env: Mapping[str, str]) -> None:
    """Refuse to serve without a key: required_keys checks the entry exists, not that it is set."""
    if not env.get("API_KEY", "").strip():
        raise RuntimeError(
            f"API_KEY is missing or empty in the Modal secret {API_KEY_SECRET!r}; this endpoint "
            f"never runs without a key. Create or replace it with:\n  {CREATE_SECRET_COMMAND}"
            " --force"
        )


def wait_until_listening(process: subprocess.Popen, port: int, poll_s: float = 1.0) -> None:
    """Block until `port` accepts a TCP connection, or raise as soon as `process` exits.

    Without this, a launcher that dies during boot leaves Modal waiting for the port until
    startup_timeout: 20 minutes of billed GPU for nothing. Raising tears the container down now.
    """
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"voxtral_rt exited with code {code} during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(poll_s)
