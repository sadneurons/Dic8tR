"""Standalone audio capture test.

Run from project root:  python tests/test_audio_standalone.py

Records 3 seconds from the default mic, saves to tests/fixtures/test_capture.wav.
Play it back to verify quality: aplay tests/fixtures/test_capture.wav
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr.audio import AudioCapture, WHISPER_SAMPLE_RATE


def main() -> None:
    print("Available input devices:")
    for dev in AudioCapture.list_devices():
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch, {int(dev['sample_rate'])}Hz)")

    cap = AudioCapture()

    print(f"\nRecording for 3 seconds (native {cap._native_sr}Hz -> {WHISPER_SAMPLE_RATE}Hz)... speak now!")
    cap.start()
    time.sleep(3)
    audio = cap.stop()

    if len(audio) == 0:
        print("ERROR: No audio captured.")
        return

    duration = len(audio) / WHISPER_SAMPLE_RATE
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    print(f"Captured: {duration:.2f}s | RMS: {rms:.4f} | Peak: {peak:.4f}")

    out_path = Path(__file__).parent / "fixtures" / "test_capture.wav"
    out_path.parent.mkdir(exist_ok=True)

    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(WHISPER_SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    print(f"Saved to: {out_path}")
    print(f"Play it:  aplay {out_path}")


if __name__ == "__main__":
    main()
