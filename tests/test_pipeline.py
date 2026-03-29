"""End-to-end pipeline test: audio -> transcribe -> postprocess -> inject.

Run from project root:  python tests/test_pipeline.py

Hold PAUSE/BREAK, F9, or RIGHT ALT to record.
Press ESC to quit. Click into a text editor before speaking.

This is a headless integration test — no GUI, no system tray.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr.audio import AudioCapture
from whispr.transcribe import WhisperTranscriber
from whispr.postprocess import postprocess
from whispr.inject import inject_text
from whispr.config import load_config, load_vocabulary, build_initial_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    # Load config and vocabulary
    config = load_config()
    vocab = load_vocabulary()
    initial_prompt = build_initial_prompt(vocab)

    # Load Whisper model
    print("Loading Whisper model...")
    transcriber = WhisperTranscriber(
        model_size=config["model_size"],
        language=config["language"],
        beam_size=config["beam_size"],
        initial_prompt=initial_prompt,
    )
    transcriber.load_model()
    print("Model ready.\n")

    # Post-processing settings
    pp_config = config["postprocessing"]
    corrections = vocab.get("corrections", {})
    expansions = vocab.get("expansions", {})
    last_text = ""  # Buffer for "scratch that"

    # Set up audio
    audio_cap = AudioCapture(device=config["audio_device"])

    # Use pynput for keyboard — import here since it's not installed yet in pyproject
    try:
        from pynput import keyboard
    except ImportError:
        print("ERROR: pynput not installed. Run: pip install pynput")
        return

    recording = False

    # Hotkey options — change this to switch trigger key
    TRIGGER_KEYS = {
        keyboard.Key.pause,         # Pause/Break
        keyboard.Key.f9,            # F9
        keyboard.KeyCode.from_vk(108),  # Right Alt (KEY_RIGHT_ALT evdev code)
    }

    def is_trigger(key: keyboard.Key | keyboard.KeyCode) -> bool:
        return key in TRIGGER_KEYS

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording
        if is_trigger(key) and not recording:
            recording = True
            audio_cap.start()
            print("[REC] Recording... (release to stop)")

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        nonlocal recording
        if key == keyboard.Key.esc:
            print("\nQuitting.")
            return False

        if is_trigger(key) and recording:
            recording = False
            audio = audio_cap.stop()

            if len(audio) == 0:
                print("[---] No audio captured")
                return

            duration = len(audio) / 16_000
            print(f"[...] Transcribing {duration:.1f}s of audio...")

            t0 = time.time()
            text = transcriber.transcribe(audio)
            elapsed = time.time() - t0

            if not text:
                print(f"[---] Empty transcription ({elapsed:.2f}s)")
                return

            print(f"[RAW] ({elapsed:.2f}s) {text}")

            # Post-process
            processed = postprocess(
                text,
                corrections=corrections,
                expansions=expansions,
                enable_punctuation=pp_config["punctuation_commands"],
                enable_editing=pp_config["editing_commands"],
                enable_corrections=pp_config["vocabulary_corrections"],
                enable_expansions=pp_config["vocabulary_expansions"],
                enable_capitalisation=pp_config["auto_capitalisation"],
            )

            # Handle editing commands
            if processed == "SCRATCH_THAT":
                print("[CMD] Scratch that — discarding last utterance")
                last_text = ""
                return
            if processed == "SCRATCH_WORD":
                print("[CMD] Scratch word — not yet wired to injection")
                return

            print(f"[TXT] {processed}")

            # Inject into focused window
            ok = inject_text(
                processed,
                clipboard_threshold=config["clipboard_threshold_chars"],
                method=config["injection_method"],
            )
            if ok:
                last_text = processed
                print("[INJ] Text injected")
            else:
                print("[ERR] Injection failed")

    print("=" * 60)
    print("  Whispr Pipeline Test")
    print("  Hold PAUSE/BREAK, F9, or RIGHT ALT to record")
    print("  Press ESC to quit")
    print("  Click into a text editor before speaking!")
    print("=" * 60)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
