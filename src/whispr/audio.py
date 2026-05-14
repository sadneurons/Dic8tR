"""Audio capture via sounddevice.

Records from the system microphone using a callback-based InputStream.
Records at the device's native sample rate and resamples to 16 kHz mono
float32 — the format Whisper expects.
"""

import logging
import threading

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
