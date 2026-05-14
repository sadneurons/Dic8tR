"""Tests for the energy-based VAD utterance segmenter in whispr.audio.

The segmenter is a pure state machine (no I/O), so we feed it synthetic
chunks of "speech" (sine wave above threshold) and "silence" (zeros) and
assert it emits exactly one utterance per speech run, with end-of-speech
detection driven by ``end_of_speech_ms``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr.audio import _UtteranceSegmenter


# Chunk size in ms — must match what the segmenter is configured with so
# silence/speech durations line up with the chunk grid.
CHUNK_MS = 20.0
# Samples per chunk at a typical 44.1kHz device — only used to make synthetic
# chunks the right *shape*; the segmenter measures duration via chunk_ms.
CHUNK_SAMPLES = int(44100 * CHUNK_MS / 1000)


def _speech_chunk(amplitude: float = 0.2) -> np.ndarray:
    """A chunk whose RMS is well above the default 0.01 threshold."""
    t = np.arange(CHUNK_SAMPLES) / 44100
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence_chunk() -> np.ndarray:
    return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


def _make_segmenter(**overrides) -> _UtteranceSegmenter:
    defaults = dict(
        rms_threshold=0.01,
        end_of_speech_ms=200,    # 10 chunks of silence to end
        min_speech_ms=40,         # 2 chunks of speech minimum
        max_utterance_ms=2_000,
        lookback_ms=40,           # 2-chunk lookback
        chunk_ms=CHUNK_MS,
    )
    defaults.update(overrides)
    return _UtteranceSegmenter(**defaults)


class TestSegmenterBasics:
    def test_silence_only_emits_nothing(self):
        seg = _make_segmenter()
        for _ in range(50):
            assert seg.process_chunk(_silence_chunk()) is None

    def test_speech_then_silence_emits_one_utterance(self):
        seg = _make_segmenter()
        # 10 chunks of speech (~200ms), then 10 chunks of silence to trigger EOS
        for _ in range(10):
            assert seg.process_chunk(_speech_chunk()) is None
        # First 9 silence chunks accumulate; the 10th hits end_of_speech_ms
        emitted = None
        for i in range(10):
            result = seg.process_chunk(_silence_chunk())
            if result is not None:
                emitted = result
                break
        assert emitted is not None, "Expected an utterance after ≥end_of_speech_ms of silence"
        # The buffer should contain the speech + the trailing silence we used
        # to detect end-of-speech (typical VAD trim is the caller's job).
        assert emitted.dtype == np.float32

    def test_short_blip_is_dropped(self):
        # A 1-chunk speech blip is below min_speech_ms (40ms / 2 chunks) and
        # must not produce an utterance.
        seg = _make_segmenter()
        seg.process_chunk(_speech_chunk())
        # Now sustained silence — should NOT emit because speech was too short
        for _ in range(20):
            assert seg.process_chunk(_silence_chunk()) is None

    def test_two_utterances_separated_by_silence(self):
        seg = _make_segmenter()
        emitted = []

        # Utterance 1: 5 speech, 12 silence (clears EOS)
        for _ in range(5):
            r = seg.process_chunk(_speech_chunk())
            assert r is None
        for _ in range(12):
            r = seg.process_chunk(_silence_chunk())
            if r is not None:
                emitted.append(r)

        # Utterance 2: 5 speech, 12 silence
        for _ in range(5):
            r = seg.process_chunk(_speech_chunk())
            assert r is None
        for _ in range(12):
            r = seg.process_chunk(_silence_chunk())
            if r is not None:
                emitted.append(r)

        assert len(emitted) == 2, f"Expected two utterances, got {len(emitted)}"

    def test_brief_silence_within_speech_is_not_an_end(self):
        # A short silence (< end_of_speech_ms) shouldn't end the utterance.
        seg = _make_segmenter()
        for _ in range(5):
            seg.process_chunk(_speech_chunk())
        # Brief silence — 5 chunks (100ms < 200ms threshold)
        for _ in range(5):
            assert seg.process_chunk(_silence_chunk()) is None
        # Resume speech, then a full silence run to end
        for _ in range(5):
            assert seg.process_chunk(_speech_chunk()) is None
        emitted = None
        for _ in range(15):
            r = seg.process_chunk(_silence_chunk())
            if r is not None:
                emitted = r
                break
        assert emitted is not None
        # The utterance should span ~all the chunks we fed during the long burst
        # (segmenter at 44.1kHz × 20ms/chunk = 882 samples/chunk)
        expected_min_samples = 12 * CHUNK_SAMPLES  # at least the speech chunks
        assert len(emitted) >= expected_min_samples


class TestSegmenterLookback:
    def test_lookback_captures_pre_speech_chunks(self):
        # With lookback_ms=40 (2 chunks), the segmenter should keep the 2 most
        # recent silence chunks at the moment speech is detected, so the first
        # phoneme isn't clipped.
        seg = _make_segmenter()
        # 3 chunks of silence (lookback buffer fills, oldest drops)
        for _ in range(3):
            seg.process_chunk(_silence_chunk())
        # First speech chunk
        seg.process_chunk(_speech_chunk())
        # End-of-speech run
        emitted = None
        for _ in range(15):
            r = seg.process_chunk(_silence_chunk())
            if r is not None:
                emitted = r
                break
        assert emitted is not None
        # The emitted buffer should include 2 lookback chunks + 1 speech +
        # ~10 trailing silence chunks. At minimum > 3 chunks worth.
        assert len(emitted) > 3 * CHUNK_SAMPLES


class TestSegmenterSafetyCap:
    def test_max_utterance_truncates(self):
        # max_utterance_ms=200 → after 10 chunks the segmenter should force-emit.
        seg = _make_segmenter(max_utterance_ms=200)
        emitted = None
        for i in range(20):
            r = seg.process_chunk(_speech_chunk())
            if r is not None:
                emitted = r
                break
        assert emitted is not None, "Expected force-emit when max_utterance_ms exceeded"


class TestSegmenterReset:
    def test_reset_clears_state(self):
        seg = _make_segmenter()
        for _ in range(5):
            seg.process_chunk(_speech_chunk())
        seg.reset()
        # After reset, sustained silence shouldn't emit anything
        for _ in range(20):
            assert seg.process_chunk(_silence_chunk()) is None
