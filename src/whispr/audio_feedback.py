"""Audio feedback sounds for record start/stop.

Generates short sine-wave beeps via sounddevice. No external audio files needed.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


def _generate_tone(freq: float, duration_ms: int, volume: float = 0.3) -> np.ndarray:
    """Generate a sine wave tone."""
    t = np.arange(int(SAMPLE_RATE * duration_ms / 1000)) / SAMPLE_RATE
    tone = volume * np.sin(2 * np.pi * freq * t).astype(np.float32)
    # Apply fade in/out to avoid clicks
    fade_samples = min(int(SAMPLE_RATE * 0.005), len(tone) // 4)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return tone


# Pre-generate tones
_TONE_START = _generate_tone(880, 80, 0.25)   # A5, short high beep
_TONE_STOP = _generate_tone(440, 120, 0.25)   # A4, slightly longer lower beep


def play_start_sound() -> None:
    """Play a short high beep indicating recording started."""
    _play_async(_TONE_START)


def play_stop_sound() -> None:
    """Play a short low beep indicating recording stopped."""
    _play_async(_TONE_STOP)


def _play_async(tone: np.ndarray) -> None:
    """Play a tone on a background thread to avoid blocking."""
    def _play():
        try:
            import sounddevice as sd
            sd.play(tone, SAMPLE_RATE, blocking=True)
        except Exception as e:
            logger.debug("Audio feedback failed: %s", e)

    threading.Thread(target=_play, daemon=True).start()
