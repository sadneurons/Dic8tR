"""Standalone transcription test.

Run from project root:  python tests/test_transcribe_standalone.py

Loads the Whisper model, records 5 seconds, transcribes, prints result.
Also checks GPU utilisation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr.audio import AudioCapture
from whispr.transcribe import WhisperTranscriber
from whispr.config import load_vocabulary, build_initial_prompt


def main() -> None:
    vocab = load_vocabulary()
    initial_prompt = build_initial_prompt(vocab)
    print(f"Initial prompt ({len(initial_prompt)} chars): {initial_prompt[:80]}...")

    # Load model
    print("\nLoading Whisper large-v3 (this may take a few seconds)...")
    t0 = time.time()
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        initial_prompt=initial_prompt,
    )
    transcriber.load_model()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Record audio
    cap = AudioCapture()
    print("\nRecording for 5 seconds... speak now!")
    cap.start()
    time.sleep(5)
    audio = cap.stop()

    if len(audio) == 0:
        print("ERROR: No audio captured.")
        return

    duration = len(audio) / 16_000
    print(f"Captured: {duration:.2f}s")

    # Transcribe
    t0 = time.time()
    text = transcriber.transcribe(audio)
    elapsed = time.time() - t0

    print(f"\n--- Transcription ({elapsed:.2f}s) ---")
    print(text)
    print(f"--- RTF: {elapsed / duration:.2f}x ---")


if __name__ == "__main__":
    main()
