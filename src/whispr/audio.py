"""Audio capture via sounddevice.

Records from the system microphone using a callback-based InputStream.
Records at the device's native sample rate and resamples to 16 kHz mono
float32 — the format Whisper expects.

Two capture modes:

  ``AudioCapture`` — push-to-talk. start() begins recording, stop() returns
  the captured buffer.

  ``ContinuousCapture`` — voice-activated continuous capture. Maintains a
  persistent input stream and fires the on_utterance callback per detected
  utterance, using RMS-energy VAD with a configurable end-of-speech timeout.
"""

import collections
import logging
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
BLOCKSIZE = 1024


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation. No extra dependencies."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.arange(n_samples) / ratio
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class AudioCapture:
    """Push-to-talk audio recorder.

    Usage:
        cap = AudioCapture()
        cap.start()        # begin recording
        audio = cap.stop() # stop and return 16kHz float32 numpy array
    """

    def __init__(
        self,
        device: int | None = None,
        rms_threshold: float = 0.005,
        silence_trim_ms: int = 200,
    ) -> None:
        self.device = device
        self.rms_threshold = rms_threshold
        self.silence_trim_ms = silence_trim_ms

        # Resolve the device's native sample rate
        try:
            dev_index = device if device is not None else sd.default.device[0]
            dev_info = sd.query_devices(dev_index)
            self._native_sr = int(dev_info["default_samplerate"])
            logger.info("Audio device: [%s] %s (%dHz)", dev_index, dev_info["name"], self._native_sr)
        except Exception as e:
            logger.warning("Failed to query device %s, falling back to 44100Hz: %s", device, e)
            self._native_sr = 44100

        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _refresh_sample_rate(self) -> None:
        """Re-query the device sample rate (device may have changed)."""
        try:
            dev_index = self.device if self.device is not None else sd.default.device[0]
            dev_info = sd.query_devices(dev_index)
            self._native_sr = int(dev_info["default_samplerate"])
        except Exception:
            pass  # keep last known rate

    def start(self) -> None:
        """Open the audio stream and begin accumulating chunks."""
        if self._recording:
            logger.warning("start() called while already recording")
            return

        self._refresh_sample_rate()
        self._chunks = []
        self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=self._native_sr,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(
                "Recording started (device=%s, native_sr=%d)",
                self.device or "default",
                self._native_sr,
            )
        except Exception as e:
            self._recording = False
            logger.error("Failed to open audio stream: %s", e)
            raise

    def stop(self) -> np.ndarray:
        """Stop recording and return captured audio as 16 kHz float32 numpy array.

        Returns an empty array if nothing was captured.
        """
        if not self._recording:
            logger.warning("stop() called while not recording")
            return np.array([], dtype=np.float32)

        self._recording = False

        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                # abort() stops immediately and waits for the callback to finish,
                # avoiding double-free when the callback races with close()
                stream.abort()
                stream.close()
            except Exception as e:
                logger.warning("Error closing audio stream: %s", e)

        with self._lock:
            if not self._chunks:
                logger.warning("No audio chunks captured")
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []

        native_duration = len(audio) / self._native_sr

        # Resample to 16 kHz for Whisper
        audio = _resample(audio, self._native_sr, WHISPER_SAMPLE_RATE)

        # Trim silence
        duration = len(audio) / WHISPER_SAMPLE_RATE
        logger.info("Recording stopped: %.2fs captured", native_duration)

        audio = self._trim_silence(audio)
        trimmed_duration = len(audio) / WHISPER_SAMPLE_RATE
        if trimmed_duration < duration:
            logger.debug(
                "Trimmed silence: %.2fs -> %.2fs", duration, trimmed_duration
            )

        return audio

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — accumulate audio chunks."""
        if not self._recording:
            return
        if status:
            logger.warning("Audio callback status: %s", status)
        with self._lock:
            self._chunks.append(indata.copy())

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence using an RMS energy threshold.

        Operates in windows of silence_trim_ms. If the entire clip is below
        threshold, return it unchanged (let Whisper decide).
        """
        if len(audio) == 0:
            return audio

        window_size = int(WHISPER_SAMPLE_RATE * self.silence_trim_ms / 1000)
        if window_size == 0 or len(audio) < window_size:
            return audio

        n_windows = len(audio) // window_size
        trimmed = audio[: n_windows * window_size]
        windows = trimmed.reshape(n_windows, window_size)
        rms = np.sqrt(np.mean(windows ** 2, axis=1))

        active = np.where(rms > self.rms_threshold)[0]
        if len(active) == 0:
            return audio

        start = active[0] * window_size
        end = min((active[-1] + 1) * window_size, len(audio))
        return audio[start:end]

    @staticmethod
    def list_devices() -> list[dict]:
        """Return a list of available audio input devices."""
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                inputs.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                })
        return inputs


# ── Continuous capture with voice-activated segmentation ─────────────


class _UtteranceSegmenter:
    """Streaming VAD-based utterance segmenter (energy-based).

    Feed audio chunks via ``process_chunk``. When sustained silence after
    speech is detected (``end_of_speech_ms``), returns the buffered utterance
    audio. Maintains a short pre-speech lookback so the first phoneme isn't
    clipped. Pure state machine — no I/O — so it's easy to unit-test.
    """

    def __init__(
        self,
        rms_threshold: float = 0.01,
        end_of_speech_ms: int = 900,
        min_speech_ms: int = 250,
        max_utterance_ms: int = 30_000,
        lookback_ms: int = 200,
        chunk_ms: float = 23.2,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.end_of_speech_ms = end_of_speech_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = max_utterance_ms
        self.chunk_ms = chunk_ms
        self._lookback_size = max(1, int(round(lookback_ms / chunk_ms)))
        self.reset()

    def reset(self) -> None:
        self._lookback: collections.deque = collections.deque(maxlen=self._lookback_size)
        self._utterance: list[np.ndarray] = []
        self._has_speech = False
        self._utterance_ms = 0.0
        self._silence_ms = 0.0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """Process one audio chunk. Returns the complete utterance audio
        (native sample rate) when end-of-speech is reached, else None.

        The returned array still needs resampling to 16 kHz by the caller.
        """
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32).flatten() ** 2)))
        is_speech = rms >= self.rms_threshold

        if not self._has_speech:
            # Pre-speech: roll a short lookback so the first phoneme isn't lost.
            self._lookback.append(chunk)
            if is_speech:
                # Promote lookback into the utterance buffer.
                self._utterance = list(self._lookback)
                self._lookback.clear()
                self._has_speech = True
                self._utterance_ms = self.chunk_ms * len(self._utterance)
                self._silence_ms = 0.0
            return None

        # In speech: every chunk goes to the utterance buffer.
        self._utterance.append(chunk)
        self._utterance_ms += self.chunk_ms
        if is_speech:
            self._silence_ms = 0.0
        else:
            self._silence_ms += self.chunk_ms

        if self._silence_ms >= self.end_of_speech_ms:
            return self._finalize()
        if self._utterance_ms >= self.max_utterance_ms:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray | None:
        # Speech duration = utterance length minus the trailing silence we
        # used to detect end-of-speech. Filter out very short bursts.
        speech_ms = self._utterance_ms - self._silence_ms
        if speech_ms < self.min_speech_ms:
            self.reset()
            return None
        audio = np.concatenate(self._utterance, axis=0).flatten()
        self.reset()
        return audio


class ContinuousCapture:
    """Voice-activated continuous audio capture.

    Opens a persistent input stream and feeds chunks through an
    ``_UtteranceSegmenter``. When a complete utterance is detected, the
    ``on_utterance`` callback is invoked with the audio resampled to 16 kHz.

    The callback runs on PortAudio's audio thread — connect it to a Qt
    signal (or otherwise hand off immediately) to avoid blocking the stream.
    """

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        device: int | None = None,
        rms_threshold: float = 0.01,
        end_of_speech_ms: int = 900,
        min_speech_ms: int = 250,
        max_utterance_ms: int = 30_000,
        lookback_ms: int = 200,
    ) -> None:
        self._on_utterance = on_utterance
        self.device = device

        try:
            dev_index = device if device is not None else sd.default.device[0]
            dev_info = sd.query_devices(dev_index)
            self._native_sr = int(dev_info["default_samplerate"])
        except Exception as e:
            logger.warning("Failed to query device %s, defaulting to 44100Hz: %s", device, e)
            self._native_sr = 44100

        chunk_ms = (BLOCKSIZE / self._native_sr) * 1000
        self._segmenter = _UtteranceSegmenter(
            rms_threshold=rms_threshold,
            end_of_speech_ms=end_of_speech_ms,
            min_speech_ms=min_speech_ms,
            max_utterance_ms=max_utterance_ms,
            lookback_ms=lookback_ms,
            chunk_ms=chunk_ms,
        )

        self._stream: sd.InputStream | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._segmenter.reset()
        self._running = True
        try:
            self._stream = sd.InputStream(
                samplerate=self._native_sr,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(
                "Continuous capture started (device=%s, native_sr=%d)",
                self.device if self.device is not None else "default",
                self._native_sr,
            )
        except Exception as e:
            self._running = False
            self._stream = None
            logger.error("Failed to open continuous audio stream: %s", e)
            raise

    def stop(self) -> None:
        self._running = False
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception as e:
                logger.warning("Error closing continuous stream: %s", e)
        self._segmenter.reset()
        logger.info("Continuous capture stopped")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if not self._running:
            return
        if status:
            logger.warning("Continuous callback status: %s", status)
        utterance = self._segmenter.process_chunk(indata.copy())
        if utterance is None:
            return
        audio_16k = _resample(utterance, self._native_sr, WHISPER_SAMPLE_RATE)
        try:
            self._on_utterance(audio_16k)
        except Exception:
            logger.exception("ContinuousCapture on_utterance callback raised")
