"""ensure_weights: make sure the model files are on disk, downloading once."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger("voxtral_rt.weights")

MARKER = ".complete"
# vLLM v0.31.0 loads the Mistral-format weights whenever consolidated*.safetensors exists
# (load_format "auto", model_loader/default_loader.py) and reads params.json for the same repos
# (config_format "auto", transformers_utils/config.py), so the Hugging Face-format
# model.safetensors is never read: skipping it saves ~9 GB. The small JSON files and
# tekken.json stay. assets/ holds demo images and a sample WAV that vLLM never opens.
IGNORE_PATTERNS = ["assets/*", "model.safetensors"]


def ensure_weights(repo_id: str, revision: str, root: Path) -> Path:
    """Return the local directory holding `repo_id` at `revision`, downloading if needed."""
    target = root / repo_id.replace("/", "--")
    marker = target / MARKER
    if marker.is_file() and marker.read_text().strip() == revision:
        log.info("weights already present in %s", target)
        return target
    log.info("downloading %s@%s into %s", repo_id, revision, target)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        ignore_patterns=IGNORE_PATTERNS,
    )
    marker.write_text(revision + "\n")
    log.info("download complete")
    return target
