from pathlib import Path

import pytest

import voxtral_rt.weights as weights
from voxtral_rt.weights import ensure_weights

REPO = "mistralai/Voxtral-Mini-4B-Realtime-Arabic"


def fake_download(calls):
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "consolidated.safetensors").write_bytes(b"weights")
        return str(local_dir)

    return snapshot_download


def test_downloads_once_then_marker_makes_it_a_noop(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(weights, "snapshot_download", fake_download(calls))

    first = ensure_weights(REPO, "rev1", tmp_path)
    second = ensure_weights(REPO, "rev1", tmp_path)

    assert first == second
    assert first.is_relative_to(tmp_path)
    assert (first / "consolidated.safetensors").is_file()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == REPO
    assert calls[0]["revision"] == "rev1"
    assert "assets/*" in calls[0]["ignore_patterns"]


def test_a_new_revision_downloads_again(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(weights, "snapshot_download", fake_download(calls))
    ensure_weights(REPO, "rev1", tmp_path)
    ensure_weights(REPO, "rev2", tmp_path)
    assert [c["revision"] for c in calls] == ["rev1", "rev2"]


def test_an_interrupted_download_leaves_no_marker(monkeypatch, tmp_path):
    def failing(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(weights, "snapshot_download", failing)
    with pytest.raises(OSError):
        ensure_weights(REPO, "rev1", tmp_path)
    assert not list(tmp_path.rglob(weights.MARKER))
