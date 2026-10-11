"""ensure_weights: make sure the model files are on disk, downloading once."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger("voxtral_rt.weights")

MARKER = ".complete"
# The repo also ships demo images and a sample WAV under assets/; vLLM never reads them.
# Until checkpoint A tells us which weight format vLLM loads, both formats are downloaded.
IGNORE_PATTERNS = ["assets/*"]


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
