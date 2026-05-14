"""Tests for whispr.transcribe — model lifecycle and transcript assembly.

The real WhisperModel is a ~3 GB CTranslate2 artefact loaded onto the GPU.
We don't load it here. Instead we mock the WhisperModel class and feed it
canned segments, so the tests exercise the wrapper's behaviour:

  - load_model passes the right kwargs (incl. the persistent download_root)
  - transcribe() concatenates segment text, strips, and never logs the
    transcript at INFO level (privacy regression check)
  - transcribe() short-circuits cleanly when the model isn't loaded
    or the audio is empty
  - reset_model() really clears the loaded model so a hot-swap reloads
"""

import logging
import threading

import numpy as np
import pytest

from whispr import transcribe as tr_mod
from whispr.transcribe import WhisperTranscriber


# ── Fake WhisperModel ─────────────────────────────────────────────────


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInfo:
    language = "en"
    language_probability = 0.97


class _FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel.

    instances captures every construction (args/kwargs) so tests can assert
    the wrapper passed the right values. transcribe() returns whatever was
    set on the class attribute ``segments_to_yield``.
    """
    instances: list[tuple[tuple, dict]] = []
    segments_to_yield: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        _FakeWhisperModel.instances.append((args, kwargs))

    def transcribe(self, audio, **kwargs):
        return (
            iter(_FakeSegment(s) for s in _FakeWhisperModel.segments_to_yield),
            _FakeInfo(),
        )


@pytest.fixture
def fake_whisper_model(monkeypatch):
    _FakeWhisperModel.instances.clear()
    _FakeWhisperModel.segments_to_yield = []
    monkeypatch.setattr(tr_mod, "WhisperModel", _FakeWhisperModel)
    yield _FakeWhisperModel
    _FakeWhisperModel.instances.clear()


# ── load_model: args and download_root ────────────────────────────────


def test_load_model_passes_size_device_compute(fake_whisper_model):
    t = WhisperTranscriber(model_size="medium", device="cpu",
                           compute_type="int8")
    t.load_model()

    assert len(fake_whisper_model.instances) == 1
    args, kwargs = fake_whisper_model.instances[0]
    assert args[0] == "medium"
    assert kwargs.get("device") == "cpu"
    assert kwargs.get("compute_type") == "int8"


def test_load_model_passes_persistent_download_root(fake_whisper_model):
    """WhisperModel is told to store cache under whispr's persistent dir
    rather than ~/.cache. This is the fix for the "model re-downloads
    every time I clean my cache" bug — a regression here would silently
    bring it back."""
    from whispr.config import models_dir
    t = WhisperTranscriber(model_size="medium")
    t.load_model()

    _, kwargs = fake_whisper_model.instances[0]
    assert kwargs.get("download_root") == str(models_dir())


# ── transcribe(): short-circuits and segment concatenation ────────────


def test_transcribe_returns_empty_when_model_not_loaded(fake_whisper_model):
    t = WhisperTranscriber()
    # No load_model() — _model is None.
    assert t.transcribe(np.zeros(16_000, dtype=np.float32)) == ""


def test_transcribe_returns_empty_for_empty_audio(fake_whisper_model):
    t = WhisperTranscriber()
    t.load_model()
    assert t.transcribe(np.array([], dtype=np.float32)) == ""


def test_transcribe_concatenates_segments_with_single_space(fake_whisper_model):
    fake_whisper_model.segments_to_yield = [" hello ", " world ", "  ok"]
    t = WhisperTranscriber()
    t.load_model()
    out = t.transcribe(np.zeros(16_000, dtype=np.float32))
    # Each segment stripped, joined with one space, outer strip.
    assert out == "hello world ok"


def test_transcribe_handles_empty_segments(fake_whisper_model):
    """Whisper sometimes emits an empty/whitespace segment. Filtering must
    happen BEFORE the join, otherwise we get double spaces."""
    fake_whisper_model.segments_to_yield = ["the patient", "", " is alert"]
    t = WhisperTranscriber()
    t.load_model()
    assert t.transcribe(np.zeros(16_000, dtype=np.float32)) == "the patient is alert"


def test_transcribe_returns_empty_on_underlying_failure(monkeypatch, caplog):
    """If the model raises mid-transcribe, return "" rather than propagate —
    the caller will treat empty as "skip injection" and keep the app alive."""
    class _BoomModel(_FakeWhisperModel):
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("CUDA OOM")

    monkeypatch.setattr(tr_mod, "WhisperModel", _BoomModel)
    t = WhisperTranscriber()
    t.load_model()
    with caplog.at_level(logging.ERROR):
        assert t.transcribe(np.zeros(16_000, dtype=np.float32)) == ""
    assert any("Transcription failed" in r.getMessage() for r in caplog.records)


# ── Privacy: transcript text must not appear at INFO level ────────────


def test_transcribe_does_not_log_text_at_info_level(fake_whisper_model, caplog):
    """journald and shell redirects typically capture INFO and above. The
    transcript itself goes to DEBUG only. A regression here would mean
    every dictation lands in /var/log/journal."""
    sensitive = "patient confided suicidal ideation"
    fake_whisper_model.segments_to_yield = [sensitive]
    t = WhisperTranscriber()
    t.load_model()
    with caplog.at_level(logging.INFO):
        t.transcribe(np.zeros(16_000, dtype=np.float32))

    info_messages = [r.getMessage() for r in caplog.records
                     if r.levelno >= logging.INFO]
    assert all(sensitive not in m for m in info_messages), (
        "transcript text leaked into INFO logs"
    )


# ── reset_model: hot-swap support ─────────────────────────────────────


def test_reset_model_clears_loaded_model(fake_whisper_model):
    t = WhisperTranscriber()
    t.load_model()
    assert t.is_loaded is True

    t.reset_model()
    assert t.is_loaded is False
    # transcribe now short-circuits (regression: the hot-swap UX depends on
    # this — otherwise the old model would still serve until GC).
    assert t.transcribe(np.zeros(16_000, dtype=np.float32)) == ""


def test_reset_model_under_lock_serialises_with_transcribe(fake_whisper_model):
    """reset_model holds the transcription lock so it doesn't pull the
    model out from under an in-flight transcribe(). Approximate this by
    holding the lock manually and asserting reset_model would block.
    Without the lock, reset_model would zero _model immediately."""
    t = WhisperTranscriber()
    t.load_model()

    # Take the lock from another thread (simulating an in-flight transcribe)
    # then try to reset. The reset must wait — it should not complete until
    # the lock is released.
    held = threading.Event()
    released = threading.Event()

    def holder():
        with t._lock:
            held.set()
            released.wait(timeout=2)

    h = threading.Thread(target=holder, daemon=True)
    h.start()
    held.wait(timeout=2)

    reset_done = threading.Event()

    def do_reset():
        t.reset_model()
        reset_done.set()

    r = threading.Thread(target=do_reset, daemon=True)
    r.start()
    # reset_model must NOT have completed while we hold the lock.
    assert not reset_done.wait(timeout=0.1)

    released.set()
    h.join(timeout=2)
    assert reset_done.wait(timeout=2)
    assert t.is_loaded is False


def test_load_after_reset_reloads(fake_whisper_model):
    """After reset, the next load_model must construct a fresh WhisperModel
    rather than no-op. This is what makes the hot-swap effective."""
    t = WhisperTranscriber()
    t.load_model()
    t.reset_model()
    t.load_model()

    assert len(fake_whisper_model.instances) == 2


# ── load_model_async: errors flow into on_error ───────────────────────


def test_load_model_async_invokes_on_complete(fake_whisper_model):
    done = threading.Event()
    errs: list[Exception] = []
    t = WhisperTranscriber()
    thread = t.load_model_async(on_complete=done.set,
                                on_error=errs.append)
    thread.join(timeout=2)
    assert done.is_set()
    assert errs == []


def test_load_model_async_invokes_on_error_on_failure(monkeypatch):
    class _BoomModel:
        def __init__(self, *a, **kw):
            raise RuntimeError("no CUDA")
    monkeypatch.setattr(tr_mod, "WhisperModel", _BoomModel)

    errs: list[Exception] = []
    t = WhisperTranscriber()
    thread = t.load_model_async(on_complete=lambda: None,
                                on_error=errs.append)
    thread.join(timeout=2)
    assert len(errs) == 1
    assert "no CUDA" in str(errs[0])
