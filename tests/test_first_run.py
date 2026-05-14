"""Tests for the model-cache integrity check.

Uses pytest's tmp_path + monkeypatch to redirect models_dir() at the source,
so the tests build a fake HF-style snapshot tree on disk and verify
model_is_cached() correctly identifies intact, partial, and missing caches.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr import first_run
from whispr.first_run import (
    _REQUIRED_MODEL_FILES,
    _snapshot_is_intact,
    model_is_cached,
)


def _build_snapshot(root: Path, model_size: str = "large-v3", *, missing: tuple = (), zero_byte: tuple = ()) -> Path:
    """Create a fake faster-whisper snapshot at root/models--<repo>/snapshots/<hash>/.

    By default every required file is present and contains a few bytes.
    Pass missing=(...) to skip files and zero_byte=(...) to write empty ones.
    """
    repo = first_run.MODEL_INFO[model_size]["repo"]
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for name in _REQUIRED_MODEL_FILES:
        if name in missing:
            continue
        path = snap / name
        if name in zero_byte:
            path.touch()
        else:
            path.write_bytes(b"x" * 16)
    return snap


@pytest.fixture
def fake_models_dir(tmp_path, monkeypatch):
    """Redirect first_run.models_dir() to a writable temp dir."""
    monkeypatch.setattr(first_run, "models_dir", lambda: tmp_path)
    return tmp_path


class TestSnapshotIsIntact:
    def test_intact_snapshot(self, tmp_path):
        snap = _build_snapshot(tmp_path)
        assert _snapshot_is_intact(snap) is True

    def test_missing_model_bin(self, tmp_path):
        snap = _build_snapshot(tmp_path, missing=("model.bin",))
        assert _snapshot_is_intact(snap) is False

    def test_zero_byte_model_bin(self, tmp_path):
        snap = _build_snapshot(tmp_path, zero_byte=("model.bin",))
        assert _snapshot_is_intact(snap) is False

    def test_missing_tokenizer(self, tmp_path):
        snap = _build_snapshot(tmp_path, missing=("tokenizer.json",))
        assert _snapshot_is_intact(snap) is False

    def test_dangling_symlink_treated_as_missing(self, tmp_path):
        snap = _build_snapshot(tmp_path)
        # Replace model.bin with a symlink to a nonexistent target — simulates
        # an HF blob that got deleted but the snapshot symlink remains.
        target = snap / "model.bin"
        target.unlink()
        target.symlink_to(tmp_path / "does-not-exist")
        assert _snapshot_is_intact(snap) is False


class TestModelIsCached:
    def test_intact_cache_returns_true(self, fake_models_dir):
        _build_snapshot(fake_models_dir)
        assert model_is_cached("large-v3") is True

    def test_no_repo_dir_returns_false(self, fake_models_dir):
        assert model_is_cached("large-v3") is False

    def test_empty_snapshots_returns_false(self, fake_models_dir):
        # Repo dir exists but the snapshots dir is empty (an interrupted
        # download can leave this state).
        repo = first_run.MODEL_INFO["large-v3"]["repo"]
        (fake_models_dir / ("models--" + repo.replace("/", "--")) / "snapshots").mkdir(parents=True)
        assert model_is_cached("large-v3") is False

    def test_partial_download_returns_false(self, fake_models_dir):
        _build_snapshot(fake_models_dir, missing=("model.bin",))
        assert model_is_cached("large-v3") is False

    def test_unknown_model_size_returns_false(self, fake_models_dir):
        _build_snapshot(fake_models_dir)
        assert model_is_cached("nonexistent-size") is False

    def test_one_partial_one_intact_returns_true(self, fake_models_dir):
        # A single repo can have multiple snapshot revisions; if any one is
        # intact, the cache is usable.
        repo = first_run.MODEL_INFO["large-v3"]["repo"]
        repo_dir = fake_models_dir / ("models--" + repo.replace("/", "--"))
        bad = repo_dir / "snapshots" / "bad"
        good = repo_dir / "snapshots" / "good"
        bad.mkdir(parents=True)
        good.mkdir(parents=True)
        # Bad: only config.json
        (bad / "config.json").write_bytes(b"x")
        # Good: all required files
        for name in _REQUIRED_MODEL_FILES:
            (good / name).write_bytes(b"x" * 16)
        assert model_is_cached("large-v3") is True
