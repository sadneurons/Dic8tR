"""Tests for AudioCapture and ContinuousCapture stream paths.

The _UtteranceSegmenter (the pure VAD state machine) already has thorough
coverage in test_audio_vad.py. This file covers the parts that wrap an
actual sounddevice.InputStream — opening/closing it, accumulating chunks
in the callback, resampling at stop, and (for ContinuousCapture) driving
the segmenter from the audio thread.

sounddevice.InputStream is replaced wholesale by FakeInputStream so the
tests don't need a real microphone.
"""

from typing import Callable

import numpy as np
import pytest

from whispr import audio as audio_mod
from whispr.audio import AudioCapture, ContinuousCapture, WHISPER_SAMPLE_RATE


# ── Fake sounddevice ──────────────────────────────────────────────────


class FakeInputStream:
    """Replacement for sounddevice.InputStream.

    Records the constructor kwargs so tests can assert samplerate/channels
    were what they expected, and stores the callback so tests can drive it
    with synthetic audio chunks. start/abort/close are bookkeeping only.
    """
    instances: list["FakeInputStream"] = []

    def __init__(self, *, samplerate, channels, dtype, blocksize, device,
                 callback: Callable, **_) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.started = False
        self.aborted = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self):
        self.started = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True

    def feed(self, chunk: np.ndarray) -> None:
        """Drive the callback with a synthetic chunk, mimicking PortAudio."""
        self.callback(chunk, len(chunk), None, 0)


@pytest.fixture
def fake_sd(monkeypatch):
    FakeInputStream.instances.clear()
    monkeypatch.setattr(audio_mod.sd, "InputStream", FakeInputStream)
    # Make device-query deterministic.
    monkeypatch.setattr(audio_mod.sd, "query_devices",
                        lambda _: {"name": "fake", "default_samplerate": 48000})
    monkeypatch.setattr(audio_mod.sd, "default", type("X", (),
                        {"device": (0, 0)})())
    yield FakeInputStream
    FakeInputStream.instances.clear()


# ── AudioCapture: start, callback, stop, resample, trim ──────────────


def test_audio_capture_opens_stream_with_native_sample_rate(fake_sd):
    cap = AudioCapture(device=0)
    cap.start()
    assert len(fake_sd.instances) == 1
    stream = fake_sd.instances[0]
    assert stream.started is True
    assert stream.samplerate == 48000
    assert stream.channels == 1
    assert stream.dtype == "float32"


def test_audio_capture_double_start_is_noop(fake_sd, caplog):
    """The hotkey thread and the tray-click slot can both race into start().
    A second start while already recording must not open a second stream."""
    cap = AudioCapture(device=0)
    cap.start()
    cap.start()
    assert len(fake_sd.instances) == 1
    assert any("already recording" in r.getMessage() for r in caplog.records)


def test_audio_capture_stop_without_start_returns_empty(fake_sd):
    cap = AudioCapture(device=0)
    out = cap.stop()
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_audio_capture_concatenates_chunks_and_resamples_to_16k(fake_sd):
    """Native rate is 48k → resampled to 16k. After feeding ~1s of audio
    at 48k, the returned array should be ~1s at 16k (≈16000 samples) minus
    whatever _trim_silence's 200 ms window truncation drops (up to 3200
    residual samples). The check is generous in the residual direction but
    strict on the upper bound: the resampler must not somehow *expand* the
    audio."""
    cap = AudioCapture(device=0)
    cap.start()
    stream = fake_sd.instances[0]
    # Four 0.25s chunks at 48k = 48000 samples = 1s. Use a constant non-zero
    # value so silence-trim treats it all as active speech.
    chunk = np.full(12_000, 0.5, dtype=np.float32).reshape(-1, 1)
    for _ in range(4):
        stream.feed(chunk)

    out = cap.stop()
    # 48000 samples @ 48 kHz → 16000 samples @ 16 kHz, minus up to one
    # 200 ms trim-window worth of remainder (3200 samples).
    assert 12_800 <= out.size <= 16_000
    assert out.dtype == np.float32


def test_audio_capture_stop_aborts_and_closes_stream(fake_sd):
    cap = AudioCapture(device=0)
    cap.start()
    stream = fake_sd.instances[0]
    cap.stop()
    assert stream.aborted is True
    assert stream.closed is True


def test_audio_capture_silence_only_returns_unchanged_array(fake_sd):
    """If the entire clip is below the RMS threshold, _trim_silence returns
    the audio as-is (let Whisper decide). The returned array must still be
    non-empty rather than zero-length, otherwise the caller treats it as
    "no audio captured"."""
    cap = AudioCapture(device=0)
    cap.start()
    stream = fake_sd.instances[0]
    # 0.5s of near-silence.
    stream.feed(np.zeros((24_000, 1), dtype=np.float32))
    out = cap.stop()
    assert out.size > 0


def test_audio_capture_callback_after_stop_does_not_accumulate(fake_sd):
    """The PortAudio callback can fire one last time after stop() flips
    _recording to False. The callback must early-return rather than push
    onto a buffer that's already been consumed."""
    cap = AudioCapture(device=0)
    cap.start()
    stream = fake_sd.instances[0]
    stream.feed(np.full((4800, 1), 0.4, dtype=np.float32))
    cap.stop()

    # Now feed AGAIN after stop. The callback should ignore it.
    stream.feed(np.full((4800, 1), 0.4, dtype=np.float32))
    # No public state to assert on except that we don't blow up.
    # _chunks should be empty because stop() drained it.
    assert cap._chunks == []


# ── list_devices ──────────────────────────────────────────────────────


def test_list_devices_filters_to_inputs(monkeypatch):
    fake_devs = [
        {"max_input_channels": 1, "name": "mic", "default_samplerate": 48000},
        {"max_input_channels": 0, "name": "speakers", "default_samplerate": 48000},
        {"max_input_channels": 2, "name": "headset", "default_samplerate": 44100},
    ]
    monkeypatch.setattr(audio_mod.sd, "query_devices", lambda: fake_devs)
    result = AudioCapture.list_devices()
    names = [d["name"] for d in result]
    assert "mic" in names and "headset" in names
    assert "speakers" not in names


# ── ContinuousCapture: callback drives the segmenter ─────────────────


def test_continuous_capture_emits_utterance_after_silence(fake_sd):
    """Feed a speech-then-silence sequence into the audio callback and
    assert on_utterance fires exactly once with non-empty audio."""
    received: list[np.ndarray] = []
    cap = ContinuousCapture(
        on_utterance=received.append,
        device=0,
        rms_threshold=0.01,
        end_of_speech_ms=100,    # short so the test runs fast
        min_speech_ms=20,
        lookback_ms=20,
    )
    cap.start()
    stream = fake_sd.instances[0]

    # The chunk size used in the segmenter calculation is BLOCKSIZE samples
    # at the native rate. Build chunks of that shape.
    chunk_len = audio_mod.BLOCKSIZE   # 1024 samples
    speech = np.full((chunk_len, 1), 0.4, dtype=np.float32)
    silence = np.zeros((chunk_len, 1), dtype=np.float32)

    # Roughly 0.5s of speech, then 0.5s of silence — enough to trigger
    # end-of-speech (100ms threshold) with margin.
    for _ in range(25):
        stream.feed(speech)
    for _ in range(25):
        stream.feed(silence)

    assert len(received) == 1
    utterance = received[0]
    assert utterance.size > 0
    # ContinuousCapture resamples to 16 kHz before invoking the callback.
    assert utterance.dtype == np.float32


def test_continuous_capture_double_start_is_noop(fake_sd):
    cap = ContinuousCapture(on_utterance=lambda _: None, device=0)
    cap.start()
    cap.start()
    assert len(fake_sd.instances) == 1


def test_continuous_capture_stop_aborts_and_closes(fake_sd):
    cap = ContinuousCapture(on_utterance=lambda _: None, device=0)
    cap.start()
    stream = fake_sd.instances[0]
    cap.stop()
    assert stream.aborted is True
    assert stream.closed is True
    assert cap.is_running is False


def test_continuous_capture_callback_exception_logged_not_propagated(fake_sd,
                                                                     caplog):
    """A buggy on_utterance callback must not crash the PortAudio thread —
    if it did, continuous mode would silently die mid-session."""
    def boom(_: np.ndarray) -> None:
        raise RuntimeError("downstream blew up")

    cap = ContinuousCapture(
        on_utterance=boom,
        device=0,
        rms_threshold=0.01,
        end_of_speech_ms=50,
        min_speech_ms=20,
        lookback_ms=20,
    )
    cap.start()
    stream = fake_sd.instances[0]
    chunk_len = audio_mod.BLOCKSIZE
    speech = np.full((chunk_len, 1), 0.4, dtype=np.float32)
    silence = np.zeros((chunk_len, 1), dtype=np.float32)
    for _ in range(20):
        stream.feed(speech)
    for _ in range(20):
        stream.feed(silence)

    # No exception propagated. The error is in the log.
    assert any("on_utterance callback raised" in r.getMessage()
               for r in caplog.records)


def test_continuous_capture_callback_after_stop_does_not_run_segmenter(fake_sd):
    """If a late audio chunk arrives after stop(), it must early-return
    rather than push into the segmenter (which has been reset)."""
    received: list[np.ndarray] = []
    cap = ContinuousCapture(on_utterance=received.append, device=0,
                            rms_threshold=0.01, end_of_speech_ms=50,
                            min_speech_ms=20, lookback_ms=20)
    cap.start()
    stream = fake_sd.instances[0]
    cap.stop()
    # Feed loud audio after stop. The callback returns early because
    # _running is False, so on_utterance never fires.
    stream.feed(np.full((audio_mod.BLOCKSIZE, 1), 0.4, dtype=np.float32))
    assert received == []
