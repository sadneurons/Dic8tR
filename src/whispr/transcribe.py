"""Whisper transcription wrapper using faster-whisper.

Loads a CTranslate2 Whisper model onto the GPU once at startup,
then transcribes numpy audio arrays on demand.
"""

import logging
import threading
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Models available in settings UI
AVAILABLE_MODELS = ("medium", "large-v3", "large-v3-turbo")


class WhisperTranscriber:
    """GPU-accelerated speech-to-text using faster-whisper.

    The model is loaded once via load_model() and reused for all
    subsequent transcriptions. Thread-safe: a lock prevents concurrent
    access to the model.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        beam_size: int = 5,
        initial_prompt: str = "",
        vad_filter: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter

        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Load the Whisper model into GPU memory. Call once at startup."""
        logger.info(
            "Loading Whisper model '%s' on %s (%s)...",
            self.model_size, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Model loaded successfully")

    def load_model_async(
        self,
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> threading.Thread:
        """Load the model in a background thread.

        Returns the thread so the caller can optionally join() it.
        """
        def _load() -> None:
            try:
                self.load_model()
                if on_complete:
                    on_complete()
            except Exception as e:
                logger.error("Model loading failed: %s", e)
                if on_error:
                    on_error(e)

        t = threading.Thread(target=_load, name="whisper-model-loader", daemon=True)
        t.start()
        return t

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 16 kHz mono audio array to text.

        Returns the concatenated text from all segments, stripped.
        Returns empty string if the model is not loaded or audio is empty.
        """
        if self._model is None:
            logger.error("transcribe() called before model is loaded")
            return ""

        if len(audio) == 0:
            logger.warning("transcribe() called with empty audio")
            return ""

        duration = len(audio) / 16_000
        logger.info("Transcribing %.2fs of audio...", duration)

        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    audio,
                    language=self.language if self.language else None,
                    beam_size=self.beam_size,
                    initial_prompt=self.initial_prompt or None,
                    vad_filter=self.vad_filter,
                    word_timestamps=False,
                )
                # Consume the generator to get all text
                text = " ".join(
                    seg.text.strip() for seg in segments
                    if seg.text
                )

            text = text.strip()
            logger.info(
                "Transcription complete (lang=%s, prob=%.2f): %s",
                info.language,
                info.language_probability,
                text[:100] + ("..." if len(text) > 100 else ""),
            )
            return text

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return ""
