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

from whispr.audio import AudioCapture, ContinuousCapture
from whispr.audio_feedback import play_start_sound, play_stop_sound
from whispr.config import load_config, load_vocabulary, load_profile, build_initial_prompt, save_config
from whispr.first_run import FirstRunWizard, model_is_cached
from whispr.hotkey import parse_hotkey, start_listener as start_hotkey_listener
from whispr.inject import (
    inject_text,
    send_undo,
    send_redo,
    send_key_combo,
    send_delete_word,
    send_backspace,
)
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

        # Continuous mode state — capture object lives only while active.
        self._continuous_active = False
        self._continuous_capture: ContinuousCapture | None = None

        self._hotkey_listener = None
        # Last successful injection — used by the "scratch that" handler.
        # _last_injection_chars counts characters actually sent (including the
        # leading space inject_text() prepends), and _last_injection_clipboard
        # records whether clipboard-paste was used. Together they let us
        # send Ctrl+Z (clipboard, atomic) vs a Backspace burst of the exact
        # length (xdotool type, one-undo-per-char in most editors).
        self._last_text = ""
        self._last_injection_chars = 0
        self._last_injection_clipboard = False

        # Shared recording flag, mutated from at least three threads: the
        # pynput hotkey listener thread, the Qt main thread (tray click), and
        # the continuous-mode QTimer dispatcher (also Qt thread). The lock
        # makes check-and-set on this flag atomic, and is held across the
        # audio start/stop calls so two concurrent triggers can't open the
        # input stream twice or close it under each other.
        self._recording = False
        self._recording_lock = threading.Lock()

        # Wire signals
        self.recording_started.connect(self._on_recording_started)
        self.recording_stopped.connect(self._on_recording_stopped)
        self.transcription_done.connect(self._on_transcription_done)
        self.streaming_update.connect(self._on_streaming_update)
        self.tray.quit_requested.connect(self._on_quit)
        self.tray.settings_requested.connect(self._on_settings_requested)
        self.tray.import_vocab_requested.connect(self._on_import_vocab)
        self.tray.profile_changed.connect(self._on_profile_changed)
        self.tray.toggle_listening.connect(self._on_tray_toggle)

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

    def _start_recording(self) -> bool:
        """Begin audio capture. Returns True on success.

        Holds the recording lock across audio.start() so a near-simultaneous
        click + hotkey press can't both pass the check and open two streams.
        """
        with self._recording_lock:
            if self._recording or not self.tray.enabled:
                return False
            try:
                if self._features.get("audio_feedback", False):
                    play_start_sound()
                self.audio.start()
                self._recording = True
            except Exception as e:
                logger.error("Failed to start recording: %s", e)
                return False
        # Emit outside the lock — slot dispatch can block briefly on the Qt
        # event loop, and we don't want unrelated callers to wait on it.
        self.recording_started.emit()
        return True

    def _stop_recording(self) -> None:
        """End audio capture and dispatch the buffer for transcription.

        Holds the recording lock across audio.stop() so a hotkey-release and a
        tray-click arriving at the same moment can't both call stop() (which
        would corrupt the shared chunk buffer in AudioCapture).
        """
        with self._recording_lock:
            if not self._recording:
                return
            self._recording = False
            if self._features.get("audio_feedback", False):
                play_stop_sound()
            audio = self.audio.stop()
        if len(audio) > 0:
            self.recording_stopped.emit(audio)
        else:
            QTimer.singleShot(0, lambda: self.tray.set_state(TrayState.IDLE))

    @pyqtSlot()
    def _on_tray_toggle(self) -> None:
        """Left-click on the tray icon: start dictation if idle, stop if listening."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_hotkey_listener(self) -> None:
        hotkey_spec = self.config.get("hotkey") or "pause"
        combo = parse_hotkey(hotkey_spec)
        if not combo:
            logger.warning(
                "Could not parse configured hotkey %r — falling back to Pause",
                hotkey_spec,
            )
            combo = parse_hotkey("pause")
            hotkey_spec = "pause"

        self._hotkey_listener = start_hotkey_listener(
            combo,
            on_combo_active=self._start_recording,
            on_combo_release=self._stop_recording,
        )
        if self._hotkey_listener is None:
            logger.error("Hotkey listener failed to start (pynput unavailable?)")
            return
        logger.info("Hotkey listener started: %s (push-to-talk)", hotkey_spec)

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
        # In continuous mode the mic is still open and listening for the next
        # utterance — return to LISTENING so the tray icon reflects that
        # rather than briefly flashing IDLE between every utterance.
        self.tray.set_state(
            TrayState.LISTENING if self._continuous_active else TrayState.IDLE
        )

        if not raw_text:
            logger.info("Empty transcription, skipping")
            return

        # Transcript text logged at DEBUG only so it does not appear in
        # default INFO-level captures (journald, shell redirects, etc.).
        logger.debug("Raw: %s", raw_text)

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

        logger.debug("Processed: %s", processed)

        if self._features.get("preview_overlay", False):
            self._preview.show_text(processed)
        else:
            self._inject(processed)

    def _handle_action(self, action: str) -> None:
        """Execute an action command returned by postprocess."""
        method = self.config["injection_method"]

        if action == "ACTION:SCRATCH_THAT":
            # Clipboard paste is one atomic edit → one Ctrl+Z reverses it.
            # xdotool type writes char-by-char and most editors record each as
            # a separate undo step, so Ctrl+Z would only drop one letter.
            # Backspace the exact length instead, which works regardless of
            # the target app's undo granularity.
            if not self._last_text:
                logger.info("Scratch that — nothing to undo")
            elif self._last_injection_clipboard:
                send_undo(method=method)
                logger.info("Scratch that — sent Ctrl+Z (clipboard paste)")
            else:
                send_backspace(self._last_injection_chars, method=method)
                logger.info(
                    "Scratch that — backspaced %d chars (typed)",
                    self._last_injection_chars,
                )
            self._last_text = ""
            self._last_injection_chars = 0
            self._last_injection_clipboard = False

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
        """Inject text into the focused window.

        Records the method and length actually used so the next "scratch that"
        can produce the right reversal — Ctrl+Z (atomic) for clipboard paste,
        or a Backspace burst of exactly the right length for xdotool type.
        """
        ok = inject_text(
            text,
            clipboard_threshold=self.config["clipboard_threshold_chars"],
            method=self.config["injection_method"],
        )
        if ok:
            self._last_text = text
            # Mirror inject_text's leading-space prepend (skipped only when the
            # text begins with a newline) so the recorded length matches what
            # was actually typed.
            prepended_len = len(text) + (
                1 if text and text[0] != "\n" else 0
            )
            self._last_injection_chars = prepended_len
            self._last_injection_clipboard = (
                prepended_len > self.config["clipboard_threshold_chars"]
            )
            self.tray.set_last_transcript(text)
        else:
            logger.error("Text injection failed")
            self.tray.show_notification("Whispr", "Injection failed", 2000)

    # ── Continuous / VAD mode ─────────────────────────────────────────

    def start_continuous_mode(self) -> None:
        """Start voice-activated continuous listening.

        Opens a persistent input stream. The ContinuousCapture's segmenter
        emits an utterance every time end-of-speech is detected; the same
        transcription pipeline as push-to-talk handles each one.
        """
        if self._continuous_active:
            return

        try:
            self._continuous_capture = ContinuousCapture(
                on_utterance=self._on_continuous_utterance,
                device=self.config.get("audio_device"),
            )
            self._continuous_capture.start()
        except Exception as e:
            logger.error("Failed to start continuous mode: %s", e)
            self._continuous_capture = None
            self.tray.show_notification(
                "Whispr", "Continuous mode unavailable — audio device error", 5000
            )
            return

        self._continuous_active = True
        self.tray.set_state(TrayState.LISTENING)
        self.tray.show_notification("Whispr", "Continuous listening active", 2000)
        logger.info("Continuous mode started (VAD-based segmentation)")

    def stop_continuous_mode(self) -> None:
        """Stop continuous listening."""
        self._continuous_active = False
        if self._continuous_capture is not None:
            try:
                self._continuous_capture.stop()
            except Exception as e:
                logger.warning("Error stopping continuous capture: %s", e)
            self._continuous_capture = None
        self.tray.set_state(TrayState.IDLE)
        logger.info("Continuous mode stopped")

    def _on_continuous_utterance(self, audio: np.ndarray) -> None:
        """Callback fired by ContinuousCapture per detected utterance.

        Runs on PortAudio's audio thread — we hand off immediately via a Qt
        queued-connection signal so the audio callback doesn't block on
        transcription.
        """
        if len(audio) > 0:
            self.recording_stopped.emit(audio)

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
        old_hotkey = self.config.get("hotkey") or "pause"
        self.config = new_config

        self._pp_config = new_config["postprocessing"]
        self._features = new_config.get("features", {})

        self.transcriber.language = new_config["language"]
        self.transcriber.beam_size = new_config["beam_size"]

        # Restart the global hotkey listener if the bound key changed —
        # parse_hotkey result is captured at start, so a config change is
        # otherwise inert until the next launch.
        new_hotkey = new_config.get("hotkey") or "pause"
        if new_hotkey != old_hotkey:
            if self._hotkey_listener is not None:
                try:
                    self._hotkey_listener.stop()
                except Exception as e:
                    logger.warning("Failed to stop old hotkey listener: %s", e)
                self._hotkey_listener = None
            self._start_hotkey_listener()
            self.tray.show_notification("Whispr", f"Hotkey: {new_hotkey}", 2000)

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
            self._hot_swap_model(new_config["model_size"])

    def _hot_swap_model(self, new_size: str) -> None:
        """Swap the active Whisper model in-place without restarting the app.

        Refuses if dictation is currently in flight (would lose audio) or if
        the requested model isn't on disk (user must run the first-run wizard
        to download it). Otherwise unloads the current model, switches to the
        new model_size, and kicks off an async reload.
        """
        if self._recording or self.tray.state == TrayState.PROCESSING:
            self.tray.show_notification(
                "Whispr",
                "Finish the current dictation before changing model.",
                3000,
            )
            return

        if not model_is_cached(new_size):
            self.tray.show_notification(
                "Whispr",
                f"Model '{new_size}' not downloaded. Open Settings → first-run wizard.",
                5000,
            )
            return

        logger.info("Hot-swapping model: %s → %s", self.transcriber.model_size, new_size)
        self.transcriber.model_size = new_size
        self.transcriber.reset_model()
        self.tray.set_state(TrayState.LOADING)
        self.tray.show_notification("Whispr", f"Loading {new_size}...", 2000)

        def _on_swap_complete() -> None:
            QTimer.singleShot(0, lambda: self.tray.set_state(TrayState.IDLE))
            QTimer.singleShot(
                0,
                lambda: self.tray.show_notification(
                    "Whispr", f"Model switched to {new_size}", 2000
                ),
            )

        def _on_swap_error(err: Exception) -> None:
            QTimer.singleShot(0, lambda: self.tray.set_state(TrayState.IDLE))
            QTimer.singleShot(
                0,
                lambda: self.tray.show_notification(
                    "Whispr Error", f"Model load failed: {err}", 5000
                ),
            )

        self.transcriber.load_model_async(
            on_complete=_on_swap_complete,
            on_error=_on_swap_error,
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
