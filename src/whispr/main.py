"""Entry point for the Whispr dictation application.

Initialises Qt, loads config, preloads the Whisper model, wires up
the hotkey listener / audio / transcription / post-processing / injection
pipeline, and runs the Qt event loop.
"""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from whispr.audio import AudioCapture
from whispr.config import load_config, load_vocabulary, build_initial_prompt, save_config
from whispr.inject import inject_text
from whispr.llm_cleanup import LLMCleanup
from whispr.postprocess import postprocess
from whispr.settings_dialog import SettingsDialog
from whispr.transcribe import WhisperTranscriber
from whispr.tray import WhisprTray, TrayState

logger = logging.getLogger("whispr")


class WhisprApp(QObject):
    """Core application object that owns all components and orchestrates
    the dictation pipeline.

    Signals are used for thread-safe communication between the hotkey
    listener thread, the transcription worker, and the Qt UI thread.
    """

    # Signals for cross-thread communication
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(np.ndarray)
    transcription_done = pyqtSignal(str)

    def __init__(self, config: dict, vocabulary: dict) -> None:
        super().__init__()

        self.config = config
        self.vocabulary = vocabulary
        self.initial_prompt = build_initial_prompt(vocabulary)

        self._corrections = vocabulary.get("corrections", {})
        self._expansions = vocabulary.get("expansions", {})
        self._pp_config = config["postprocessing"]

        # Components — initialised but not started
        self.tray = WhisprTray()
        self.audio = AudioCapture(device=config["audio_device"])
        self.transcriber = WhisperTranscriber(
            model_size=config["model_size"],
            language=config["language"],
            beam_size=config["beam_size"],
            initial_prompt=self.initial_prompt,
        )

        # Optional local LLM cleanup
        self._llm: LLMCleanup | None = None
        if config["postprocessing"].get("llm_cleanup", False):
            self._init_llm_cleanup()

        self._hotkey_listener = None
        self._last_text = ""

        # Wire signals
        self.recording_started.connect(self._on_recording_started)
        self.recording_stopped.connect(self._on_recording_stopped)
        self.transcription_done.connect(self._on_transcription_done)
        self.tray.quit_requested.connect(self._on_quit)
        self.tray.settings_requested.connect(self._on_settings_requested)

    def _init_llm_cleanup(self) -> None:
        """Initialise the local LLM cleanup module if enabled."""
        llm_config = self.config.get("llm", {})
        self._llm = LLMCleanup(
            model=llm_config.get("model", "mistral"),
            endpoint=llm_config.get("endpoint", "http://localhost:11434"),
        )
        if self._llm.check_availability():
            logger.info("LLM cleanup enabled (local Ollama)")
        else:
            logger.warning("LLM cleanup requested but Ollama unavailable — disabled")
            self._llm = None

    def start(self) -> None:
        """Show tray, load model async, then start hotkey listener."""
        self.tray.show()
        self.tray.set_state(TrayState.LOADING)

        # Load model in background thread
        self.transcriber.load_model_async(
            on_complete=self._on_model_loaded,
            on_error=self._on_model_error,
        )

    def _on_model_loaded(self) -> None:
        """Called from model loader thread when model is ready."""
        logger.info("Model loaded, starting hotkey listener")
        # Schedule UI update on the Qt thread
        QTimer.singleShot(0, self._activate)

    def _on_model_error(self, error: Exception) -> None:
        logger.error("Failed to load model: %s", error)
        QTimer.singleShot(
            0,
            lambda: self.tray.show_notification(
                "Whispr Error",
                f"Model failed to load: {error}",
                10000,
            ),
        )

    def _activate(self) -> None:
        """Activate the app after model is loaded (runs on Qt thread)."""
        self.tray.set_state(TrayState.IDLE)
        self.tray.show_notification("Whispr", "Ready — hold hotkey to dictate", 2000)
        self._start_hotkey_listener()

    def _start_hotkey_listener(self) -> None:
        """Start the pynput global hotkey listener."""
        try:
            from pynput import keyboard
        except ImportError:
            logger.error("pynput not installed")
            return

        recording = False
        trigger_keys = {
            keyboard.Key.pause,
            keyboard.Key.f9,
        }

        def is_trigger(key):
            return key in trigger_keys

        def on_press(key):
            nonlocal recording
            if is_trigger(key) and not recording and self.tray.enabled:
                recording = True
                self.recording_started.emit()

        def on_release(key):
            nonlocal recording
            if is_trigger(key) and recording:
                recording = False
                audio = self.audio.stop()
                if len(audio) > 0:
                    self.recording_stopped.emit(audio)
                else:
                    # No audio — go back to idle
                    QTimer.singleShot(0, lambda: self.tray.set_state(TrayState.IDLE))

        self._hotkey_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        logger.info("Hotkey listener started (Pause/Break, F9)")

    @pyqtSlot()
    def _on_recording_started(self) -> None:
        """Handle recording start (Qt thread)."""
        self.tray.set_state(TrayState.LISTENING)
        try:
            self.audio.start()
        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            self.tray.set_state(TrayState.IDLE)
            self.tray.show_notification("Whispr", f"Recording failed: {e}", 3000)

    @pyqtSlot(np.ndarray)
    def _on_recording_stopped(self, audio: np.ndarray) -> None:
        """Handle recording stop — start transcription in worker thread."""
        self.tray.set_state(TrayState.PROCESSING)

        def _worker():
            text = self.transcriber.transcribe(audio)
            # LLM cleanup runs in this worker thread (not the UI thread)
            if self._llm and self._pp_config.get("llm_cleanup", False):
                text = self._llm.cleanup(text)
            self.transcription_done.emit(text)

        t = threading.Thread(target=_worker, name="whispr-transcribe", daemon=True)
        t.start()

    @pyqtSlot(str)
    def _on_transcription_done(self, raw_text: str) -> None:
        """Handle transcription result — postprocess and inject (Qt thread)."""
        self.tray.set_state(TrayState.IDLE)

        if not raw_text:
            logger.info("Empty transcription, skipping")
            return

        logger.info("Raw: %s", raw_text)

        processed = postprocess(
            raw_text,
            corrections=self._corrections,
            expansions=self._expansions,
            enable_punctuation=self._pp_config["punctuation_commands"],
            enable_editing=self._pp_config["editing_commands"],
            enable_corrections=self._pp_config["vocabulary_corrections"],
            enable_expansions=self._pp_config["vocabulary_expansions"],
            enable_capitalisation=self._pp_config["auto_capitalisation"],
        )

        # Handle editing commands
        if processed == "SCRATCH_THAT":
            logger.info("Scratch that — discarding last utterance")
            self._last_text = ""
            return
        if processed == "SCRATCH_WORD":
            logger.info("Scratch word — not yet implemented for injection")
            return

        logger.info("Processed: %s", processed)

        # Inject into focused window
        ok = inject_text(
            processed,
            clipboard_threshold=self.config["clipboard_threshold_chars"],
            method=self.config["injection_method"],
        )

        if ok:
            self._last_text = processed
            self.tray.set_last_transcript(processed)
        else:
            logger.error("Text injection failed")
            self.tray.show_notification("Whispr", "Injection failed", 2000)

    @pyqtSlot()
    def _on_settings_requested(self) -> None:
        """Open the settings dialog."""
        dialog = SettingsDialog(self.config)
        dialog.settings_changed.connect(self._apply_settings)
        dialog.exec()

    def _apply_settings(self, new_config: dict) -> None:
        """Apply changed settings and save to disk."""
        old_model = self.config.get("model_size")
        self.config = new_config

        # Update post-processing config references
        self._pp_config = new_config["postprocessing"]

        # Update transcriber settings that don't need a model reload
        self.transcriber.language = new_config["language"]
        self.transcriber.beam_size = new_config["beam_size"]

        # Save to disk
        save_config(new_config)
        logger.info("Settings saved")

        # Notify if model changed (requires restart)
        if new_config["model_size"] != old_model:
            self.tray.show_notification(
                "Whispr",
                f"Model changed to {new_config['model_size']}. Restart Whispr to apply.",
                5000,
            )

    def _on_quit(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down")
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        self.tray.hide()
        QApplication.instance().quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="whispr",
        description="Local-only Linux dictation using faster-whisper",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: ~/.config/whispr/config.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Whispr starting up")

    # Load configuration
    config = load_config(args.config)
    vocabulary = load_vocabulary()

    logger.info("Model: %s | Language: %s", config["model_size"], config["language"])

    # Create Qt application
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running when no windows open
    app.setApplicationName("Whispr")

    # Allow Ctrl+C to kill the app
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Create and start the Whispr app
    whispr = WhisprApp(config, vocabulary)
    whispr.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
