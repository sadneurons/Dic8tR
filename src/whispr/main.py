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
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from whispr.audio import AudioCapture
from whispr.audio_feedback import play_start_sound, play_stop_sound
from whispr.config import load_config, load_vocabulary, load_profile, build_initial_prompt, save_config
from whispr.first_run import FirstRunWizard, model_is_cached
from whispr.inject import inject_text, send_undo, send_redo, send_key_combo, send_delete_word
from whispr.llm_cleanup import LLMCleanup
from whispr.postprocess import postprocess
from whispr.preview_overlay import PreviewOverlay
from whispr.settings_dialog import SettingsDialog
from whispr.transcribe import WhisperTranscriber
from whispr.tray import WhisprTray, TrayState
from whispr.vocab_import import VocabImportDialog

logger = logging.getLogger("whispr")


class WhisprApp(QObject):
    """Core application object that owns all components and orchestrates
    the dictation pipeline.
    """

    # Signals for cross-thread communication
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(np.ndarray)
    transcription_done = pyqtSignal(str)
    streaming_update = pyqtSignal(str)

    def __init__(self, config: dict, vocabulary: dict) -> None:
        super().__init__()

        self.config = config
        self.vocabulary = vocabulary
        self.initial_prompt = build_initial_prompt(vocabulary)

        self._corrections = vocabulary.get("corrections", {})
        self._expansions = vocabulary.get("expansions", {})
        self._custom_commands = vocabulary.get("commands", {})
        self._pp_config = config["postprocessing"]
        self._features = config.get("features", {})

        # Components
        active_profile = config.get("vocabulary_profile", "medical")
        self.tray = WhisprTray(active_profile=active_profile)
        self.audio = AudioCapture(device=config["audio_device"])
        self.transcriber = WhisperTranscriber(
            model_size=config["model_size"],
            language=config["language"],
            beam_size=config["beam_size"],
            initial_prompt=self.initial_prompt,
        )

        # Preview overlay (created once, shown/hidden as needed)
        self._preview = PreviewOverlay()
        self._preview.accepted.connect(self._on_preview_accepted)
        self._preview.dismissed.connect(self._on_preview_dismissed)

        # Optional local LLM cleanup
        self._llm: LLMCleanup | None = None
        if config["postprocessing"].get("llm_cleanup", False):
            self._init_llm_cleanup()

        # Continuous mode state
        self._continuous_active = False
        self._vad_timer: QTimer | None = None

        self._hotkey_listener = None
        self._last_text = ""

        # Wire signals
        self.recording_started.connect(self._on_recording_started)
        self.recording_stopped.connect(self._on_recording_stopped)
        self.transcription_done.connect(self._on_transcription_done)
        self.streaming_update.connect(self._on_streaming_update)
        self.tray.quit_requested.connect(self._on_quit)
        self.tray.settings_requested.connect(self._on_settings_requested)
        self.tray.import_vocab_requested.connect(self._on_import_vocab)
        self.tray.profile_changed.connect(self._on_profile_changed)

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

        self.transcriber.load_model_async(
            on_complete=self._on_model_loaded,
            on_error=self._on_model_error,
        )

    def _on_model_loaded(self) -> None:
        logger.info("Model loaded, starting hotkey listener")
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

    # ── Hotkey listener ───────────────────────────────────────────────

    def _start_hotkey_listener(self) -> None:
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
                try:
                    if self._features.get("audio_feedback", False):
                        play_start_sound()
                    self.audio.start()
                    self.recording_started.emit()
                except Exception as e:
                    recording = False
                    logger.error("Failed to start recording: %s", e)

        def on_release(key):
            nonlocal recording
            if is_trigger(key) and recording:
                recording = False
                if self._features.get("audio_feedback", False):
                    play_stop_sound()
                audio = self.audio.stop()
                if len(audio) > 0:
                    self.recording_stopped.emit(audio)
                else:
                    QTimer.singleShot(0, lambda: self.tray.set_state(TrayState.IDLE))

        self._hotkey_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        logger.info("Hotkey listener started (Pause/Break, F9)")

    # ── Recording handlers ────────────────────────────────────────────

    @pyqtSlot()
    def _on_recording_started(self) -> None:
        self.tray.set_state(TrayState.LISTENING)

    @pyqtSlot(np.ndarray)
    def _on_recording_stopped(self, audio: np.ndarray) -> None:
        self.tray.set_state(TrayState.PROCESSING)

        use_streaming = self._features.get("streaming_transcription", False)

        def _worker():
            if use_streaming:
                self._transcribe_streaming(audio)
            else:
                text = self.transcriber.transcribe(audio)
                if self._llm and self._pp_config.get("llm_cleanup", False):
                    text = self._llm.cleanup(text)
                self.transcription_done.emit(text)

        t = threading.Thread(target=_worker, name="whispr-transcribe", daemon=True)
        t.start()

    def _transcribe_streaming(self, audio: np.ndarray) -> None:
        """Transcribe with streaming — emit partial results as segments complete."""
        if self.transcriber._model is None:
            self.transcription_done.emit("")
            return

        try:
            with self.transcriber._lock:
                segments, info = self.transcriber._model.transcribe(
                    audio,
                    language=self.transcriber.language if self.transcriber.language else None,
                    beam_size=self.transcriber.beam_size,
                    initial_prompt=self.transcriber.initial_prompt or None,
                    vad_filter=self.transcriber.vad_filter,
                    word_timestamps=False,
                )

                parts = []
                for seg in segments:
                    if seg.text:
                        parts.append(seg.text.strip())
                        # Emit partial result for live preview
                        self.streaming_update.emit(" ".join(parts))

                text = " ".join(parts).strip()

            if self._llm and self._pp_config.get("llm_cleanup", False):
                text = self._llm.cleanup(text)

            self.transcription_done.emit(text)

        except Exception as e:
            logger.error("Streaming transcription failed: %s", e)
            self.transcription_done.emit("")

    @pyqtSlot(str)
    def _on_streaming_update(self, partial_text: str) -> None:
        """Handle partial streaming results — update preview overlay if visible."""
        if self._features.get("preview_overlay", False) and self._features.get("streaming_transcription", False):
            # Show live updating preview
            if not self._preview.isVisible():
                self._preview.show_text(partial_text)
            else:
                self._preview._text_edit.setPlainText(partial_text)

    # ── Transcription result ──────────────────────────────────────────

    @pyqtSlot(str)
    def _on_transcription_done(self, raw_text: str) -> None:
        self.tray.set_state(TrayState.IDLE)

        if not raw_text:
            logger.info("Empty transcription, skipping")
            return

        logger.info("Raw: %s", raw_text)

        processed = postprocess(
            raw_text,
            corrections=self._corrections,
            expansions=self._expansions,
            custom_commands=self._custom_commands,
            enable_punctuation=self._pp_config["punctuation_commands"],
            enable_editing=self._pp_config["editing_commands"],
            enable_corrections=self._pp_config["vocabulary_corrections"],
            enable_expansions=self._pp_config["vocabulary_expansions"],
            enable_capitalisation=self._pp_config["auto_capitalisation"],
        )

        # Handle action commands
        if processed.startswith("ACTION:"):
            self._handle_action(processed)
            return

        logger.info("Processed: %s", processed)

        if self._features.get("preview_overlay", False):
            self._preview.show_text(processed)
        else:
            self._inject(processed)

    def _handle_action(self, action: str) -> None:
        """Execute an action command returned by postprocess."""
        method = self.config["injection_method"]

        if action == "ACTION:SCRATCH_THAT":
            # Undo the last injection by sending Ctrl+Z
            if self._last_text:
                # Estimate number of undos needed (one per character for xdotool type,
                # or one for clipboard paste). Send a single Ctrl+Z which undoes
                # the last atomic operation in most editors.
                send_undo(method=method)
                logger.info("Scratch that — sent undo")
                self._last_text = ""
            else:
                logger.info("Scratch that — nothing to undo")

        elif action == "ACTION:SCRATCH_WORD":
            send_delete_word(method=method)
            logger.info("Scratch word — sent Ctrl+Backspace")

        elif action == "ACTION:UNDO":
            send_undo(method=method)
            logger.info("Undo")

        elif action == "ACTION:REDO":
            send_redo(method=method)
            logger.info("Redo")

        elif action.startswith("ACTION:KEY:"):
            # Key combo(s) — may be comma-separated for sequences
            combos = action[len("ACTION:KEY:"):].split(",")
            for combo in combos:
                send_key_combo(combo.strip(), method=method)
                time.sleep(0.02)
            logger.info("Key combo: %s", action[len("ACTION:KEY:"):])

        elif action.startswith("ACTION:INSERT:"):
            # Insert literal text (used by custom commands)
            text = action[len("ACTION:INSERT:"):]
            # Expand placeholders
            from datetime import date
            text = text.replace("{{DATE}}", date.today().strftime("%d/%m/%Y"))
            text = text.replace("{{DATE_LONG}}", date.today().strftime("%d %B %Y"))
            self._inject(text)

        else:
            logger.warning("Unknown action: %s", action)

    def _on_preview_accepted(self, text: str) -> None:
        """User accepted text from preview overlay (possibly edited)."""
        self._inject(text)

    def _on_preview_dismissed(self) -> None:
        """User dismissed preview overlay."""
        logger.info("Preview dismissed, text not injected")

    def _inject(self, text: str) -> None:
        """Inject text into the focused window."""
        ok = inject_text(
            text,
            clipboard_threshold=self.config["clipboard_threshold_chars"],
            method=self.config["injection_method"],
        )
        if ok:
            self._last_text = text
            self.tray.set_last_transcript(text)
        else:
            logger.error("Text injection failed")
            self.tray.show_notification("Whispr", "Injection failed", 2000)

    # ── Continuous / VAD mode ─────────────────────────────────────────

    def start_continuous_mode(self) -> None:
        """Start continuous listening with VAD-based segmentation."""
        if self._continuous_active:
            return

        self._continuous_active = True
        logger.info("Continuous mode started")
        self.tray.show_notification("Whispr", "Continuous listening active", 2000)
        self._continuous_record_cycle()

    def stop_continuous_mode(self) -> None:
        """Stop continuous listening."""
        self._continuous_active = False
        if self.audio.is_recording:
            audio = self.audio.stop()
            if len(audio) > 0:
                self.recording_stopped.emit(audio)
        self.tray.set_state(TrayState.IDLE)
        logger.info("Continuous mode stopped")

    def _continuous_record_cycle(self) -> None:
        """Record a segment, transcribe, then start the next segment."""
        if not self._continuous_active:
            return

        try:
            self.audio.start()
            self.tray.set_state(TrayState.LISTENING)
        except Exception as e:
            logger.error("Continuous mode audio start failed: %s", e)
            self._continuous_active = False
            return

        # Record for 5 seconds then process (VAD will trim silence)
        QTimer.singleShot(5000, self._continuous_segment_done)

    def _continuous_segment_done(self) -> None:
        """Handle end of a continuous mode recording segment."""
        if not self._continuous_active:
            return

        audio = self.audio.stop()
        if len(audio) > 0:
            self.recording_stopped.emit(audio)

        # Start next segment after a short gap
        if self._continuous_active:
            QTimer.singleShot(200, self._continuous_record_cycle)

    # ── Profile switching ─────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_profile_changed(self, name: str) -> None:
        self.vocabulary = load_profile(name)
        self._corrections = self.vocabulary.get("corrections", {})
        self._expansions = self.vocabulary.get("expansions", {})
        self._custom_commands = self.vocabulary.get("commands", {})
        self.initial_prompt = build_initial_prompt(self.vocabulary)
        self.transcriber.initial_prompt = self.initial_prompt

        self.config["vocabulary_profile"] = name
        save_config(self.config)

        logger.info("Switched to profile: %s (%d corrections, %d expansions)",
                     name, len(self._corrections), len(self._expansions))
        self.tray.show_notification("Whispr", f"Vocabulary: {name.capitalize()}", 2000)

    # ── Vocab import ──────────────────────────────────────────────────

    @pyqtSlot()
    def _on_import_vocab(self) -> None:
        dialog = VocabImportDialog()
        if dialog.exec():
            self.vocabulary = load_vocabulary()
            self._corrections = self.vocabulary.get("corrections", {})
            self._expansions = self.vocabulary.get("expansions", {})
            self._custom_commands = self.vocabulary.get("commands", {})
            self.initial_prompt = build_initial_prompt(self.vocabulary)
            self.transcriber.initial_prompt = self.initial_prompt
            logger.info("Vocabulary reloaded after import")
            self.tray.show_notification("Whispr", "Vocabulary imported successfully", 2000)

    # ── Settings ──────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_settings_requested(self) -> None:
        dialog = SettingsDialog(self.config)
        dialog.settings_changed.connect(self._apply_settings)
        dialog.exec()

    def _apply_settings(self, new_config: dict) -> None:
        old_model = self.config.get("model_size")
        old_profile = self.config.get("vocabulary_profile")
        self.config = new_config

        self._pp_config = new_config["postprocessing"]
        self._features = new_config.get("features", {})

        self.transcriber.language = new_config["language"]
        self.transcriber.beam_size = new_config["beam_size"]

        # If profile changed via settings dialog, apply it
        new_profile = new_config.get("vocabulary_profile")
        if new_profile and new_profile != old_profile:
            self._on_profile_changed(new_profile)
            self.tray.set_active_profile(new_profile)

        # Handle continuous mode toggle
        if self._features.get("continuous_mode", False) and not self._continuous_active:
            self.start_continuous_mode()
        elif not self._features.get("continuous_mode", False) and self._continuous_active:
            self.stop_continuous_mode()

        save_config(new_config)
        logger.info("Settings saved")

        if new_config["model_size"] != old_model:
            self.tray.show_notification(
                "Whispr",
                f"Model changed to {new_config['model_size']}. Restart Whispr to apply.",
                5000,
            )

    # ── Quit ──────────────────────────────────────────────────────────

    def _on_quit(self) -> None:
        logger.info("Shutting down")
        self._continuous_active = False
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        self._preview.close()
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

    config = load_config(args.config)
    profile_name = config.get("vocabulary_profile", "medical")
    vocabulary = load_profile(profile_name)
    logger.info("Vocabulary profile: %s", profile_name)
    logger.info("Model: %s | Language: %s", config["model_size"], config["language"])

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Whispr")

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if not model_is_cached(config["model_size"]):
        logger.info("Model not cached, showing first-run wizard")
        from PyQt6.QtWidgets import QDialog
        wizard = FirstRunWizard(config)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            logger.info("First-run wizard cancelled")
            return 0
        config["model_size"] = wizard.selected_model
        save_config(config)

    whispr = WhisprApp(config, vocabulary)
    whispr.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
