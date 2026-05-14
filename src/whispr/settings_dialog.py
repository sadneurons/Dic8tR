"""Settings dialog for Whispr.

Qt dialog for configuring model size, language, hotkey, audio device,
post-processing toggles, and LLM cleanup options.
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from whispr.audio import AudioCapture
from whispr.config import list_profiles
from whispr.hotkey import parse_hotkey
from whispr.transcribe import AVAILABLE_MODELS

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Configuration dialog for Whispr.

    Emits settings_changed(dict) with the updated config when accepted.
    """

    settings_changed = pyqtSignal(dict)

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Whispr Settings")
        self.setMinimumWidth(450)
        self._config = config

        layout = QVBoxLayout(self)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_transcription_tab(), "Transcription")
        tabs.addTab(self._build_audio_tab(), "Audio")
        tabs.addTab(self._build_postprocessing_tab(), "Post-Processing")
        tabs.addTab(self._build_features_tab(), "Features")
        tabs.addTab(self._build_hotkey_tab(), "Hotkey")
        tabs.addTab(self._build_llm_tab(), "LLM Cleanup")
        layout.addWidget(tabs)

        # Restart notice
        self._restart_label = QLabel("")
        self._restart_label.setStyleSheet("color: #E65100; font-style: italic;")
        self._restart_label.setVisible(False)
        layout.addWidget(self._restart_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # --- Tab builders ---

    def _build_transcription_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        # Model size
        self._model_combo = QComboBox()
        for model in AVAILABLE_MODELS:
            self._model_combo.addItem(model)
        current_model = self._config.get("model_size", "large-v3")
        idx = self._model_combo.findText(current_model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.currentTextChanged.connect(self._flag_restart)
        form.addRow("Model:", self._model_combo)

        # Language
        self._language_edit = QLineEdit(self._config.get("language", "en"))
        self._language_edit.setPlaceholderText("en (or blank for auto-detect)")
        self._language_edit.setMaximumWidth(100)
        form.addRow("Language:", self._language_edit)

        # Beam size
        self._beam_spin = QSpinBox()
        self._beam_spin.setRange(1, 20)
        self._beam_spin.setValue(self._config.get("beam_size", 5))
        form.addRow("Beam size:", self._beam_spin)

        # Injection method
        self._inject_combo = QComboBox()
        self._inject_combo.addItems(["auto", "x11", "wayland"])
        current_method = self._config.get("injection_method", "auto")
        idx = self._inject_combo.findText(current_method)
        if idx >= 0:
            self._inject_combo.setCurrentIndex(idx)
        form.addRow("Injection method:", self._inject_combo)

        # Clipboard threshold
        self._clipboard_spin = QSpinBox()
        self._clipboard_spin.setRange(0, 10000)
        self._clipboard_spin.setSingleStep(100)
        self._clipboard_spin.setValue(self._config.get("clipboard_threshold_chars", 500))
        self._clipboard_spin.setSuffix(" chars")
        form.addRow("Clipboard paste above:", self._clipboard_spin)

        return widget

    def _build_audio_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        # Audio device
        self._device_combo = QComboBox()
        self._device_combo.addItem("System default", None)

        try:
            devices = AudioCapture.list_devices()
        except Exception:
            devices = []
        current_device = self._config.get("audio_device")
        selected_idx = 0

        for dev in devices:
            label = f"[{dev['index']}] {dev['name']} ({dev['channels']}ch, {int(dev['sample_rate'])}Hz)"
            self._device_combo.addItem(label, dev["index"])
            if dev["index"] == current_device:
                selected_idx = self._device_combo.count() - 1

        self._device_combo.setCurrentIndex(selected_idx)
        form.addRow("Input device:", self._device_combo)

        return widget

    def _build_postprocessing_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Vocabulary profile selector
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Vocabulary Profile:"))
        self._profile_combo = QComboBox()
        current_profile = self._config.get("vocabulary_profile", "medical")
        for name in list_profiles():
            self._profile_combo.addItem(name.capitalize(), name)
        idx = self._profile_combo.findData(current_profile)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        profile_row.addWidget(self._profile_combo, 1)
        layout.addLayout(profile_row)

        layout.addSpacing(8)

        pp = self._config.get("postprocessing", {})

        self._pp_punctuation = QCheckBox("Punctuation commands (full stop, comma, etc.)")
        self._pp_punctuation.setChecked(pp.get("punctuation_commands", True))
        layout.addWidget(self._pp_punctuation)

        self._pp_editing = QCheckBox("Editing commands (scratch that, scratch word)")
        self._pp_editing.setChecked(pp.get("editing_commands", True))
        layout.addWidget(self._pp_editing)

        self._pp_corrections = QCheckBox("Vocabulary corrections")
        self._pp_corrections.setChecked(pp.get("vocabulary_corrections", True))
        layout.addWidget(self._pp_corrections)

        self._pp_expansions = QCheckBox("Vocabulary expansions")
        self._pp_expansions.setChecked(pp.get("vocabulary_expansions", True))
        layout.addWidget(self._pp_expansions)

        self._pp_capitalisation = QCheckBox("Auto-capitalisation")
        self._pp_capitalisation.setChecked(pp.get("auto_capitalisation", True))
        layout.addWidget(self._pp_capitalisation)

        layout.addStretch()

        # Vocabulary import button
        from PyQt6.QtWidgets import QHBoxLayout as _HBox
        vocab_row = _HBox()
        vocab_label = QLabel(
            "Vocabulary file: ~/.config/whispr/vocabulary.json"
        )
        vocab_label.setStyleSheet("color: #777; font-size: 11px;")
        vocab_row.addWidget(vocab_label, 1)
        import_btn = QPushButton("Import...")
        import_btn.clicked.connect(self._on_import_vocab)
        vocab_row.addWidget(import_btn)
        layout.addLayout(vocab_row)

        return widget

    def _build_hotkey_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        # Hotkey mode
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["push_to_talk", "toggle"])
        current_mode = self._config.get("hotkey_mode", "push_to_talk")
        idx = self._mode_combo.findText(current_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        form.addRow("Mode:", self._mode_combo)

        # Configurable hotkey — parsed by whispr.hotkey.parse_hotkey at startup.
        self._hotkey_edit = QLineEdit(self._config.get("hotkey") or "pause")
        self._hotkey_edit.setPlaceholderText("pause")
        self._hotkey_edit.textChanged.connect(self._validate_hotkey)
        form.addRow("Trigger key:", self._hotkey_edit)

        # Live validation status — turns red on an unparseable combo so the
        # user gets feedback before clicking OK and falling back to "pause".
        self._hotkey_status = QLabel("")
        self._hotkey_status.setStyleSheet("color: #777; font-size: 11px;")
        form.addRow("", self._hotkey_status)
        self._validate_hotkey(self._hotkey_edit.text())

        note = QLabel(
            "Examples: pause, f9, ctrl+shift+space, ctrl+alt+v.\n"
            "Modifiers: ctrl, shift, alt, cmd (also super/win/meta).\n"
            "Unparseable values fall back to Pause."
        )
        note.setStyleSheet("color: #777; font-size: 11px;")
        form.addRow("", note)

        return widget

    def _validate_hotkey(self, text: str) -> None:
        """Live-validate the hotkey edit and update the status label."""
        spec = (text or "").strip()
        if not spec:
            self._hotkey_status.setText("")
            return
        if parse_hotkey(spec) is not None:
            self._hotkey_status.setStyleSheet("color: #2E7D32; font-size: 11px;")
            self._hotkey_status.setText(f"✓ “{spec}” parses cleanly")
        else:
            self._hotkey_status.setStyleSheet("color: #C62828; font-size: 11px;")
            self._hotkey_status.setText(
                f"⚠ “{spec}” won’t parse — will fall back to Pause"
            )

    def _build_features_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        features = self._config.get("features", {})

        self._feat_audio_feedback = QCheckBox("Audio feedback (beep on record start/stop)")
        self._feat_audio_feedback.setChecked(features.get("audio_feedback", True))
        layout.addWidget(self._feat_audio_feedback)

        self._feat_preview = QCheckBox("Preview overlay (review/edit text before injection)")
        self._feat_preview.setChecked(features.get("preview_overlay", False))
        layout.addWidget(self._feat_preview)

        self._feat_continuous = QCheckBox("Continuous mode (always-on VAD listening, no hotkey needed)")
        self._feat_continuous.setChecked(features.get("continuous_mode", False))
        layout.addWidget(self._feat_continuous)

        self._feat_streaming = QCheckBox("Streaming transcription (show partial results while processing)")
        self._feat_streaming.setChecked(features.get("streaming_transcription", False))
        layout.addWidget(self._feat_streaming)

        layout.addSpacing(12)

        notes = QLabel(
            "Audio feedback: short beep when recording starts/stops.\n\n"
            "Preview overlay: floating window shows text before injection.\n"
            "Press Enter to inject, Esc to discard, or edit the text.\n\n"
            "Continuous mode: listens continuously using voice activity\n"
            "detection. No need to hold a hotkey — just speak.\n\n"
            "Streaming: shows partial transcription results as segments\n"
            "complete, instead of waiting for the full result."
        )
        notes.setWordWrap(True)
        notes.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(notes)

        layout.addStretch()
        return widget

    def _build_llm_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        pp = self._config.get("postprocessing", {})
        llm = self._config.get("llm", {})

        self._llm_enabled = QCheckBox("Enable local LLM cleanup (requires Ollama)")
        self._llm_enabled.setChecked(pp.get("llm_cleanup", False))
        layout.addWidget(self._llm_enabled)

        group = QGroupBox("Ollama Configuration")
        form = QFormLayout(group)

        self._llm_model = QLineEdit(llm.get("model", "mistral"))
        form.addRow("Model:", self._llm_model)

        self._llm_endpoint = QLineEdit(llm.get("endpoint", "http://localhost:11434"))
        form.addRow("Endpoint:", self._llm_endpoint)

        layout.addWidget(group)

        note = QLabel(
            "All LLM processing is strictly local via Ollama.\n"
            "No data leaves this device. Install Ollama and pull a model first:\n"
            "  ollama pull mistral"
        )
        note.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(note)

        layout.addStretch()
        return widget

    # --- Actions ---

    def _on_import_vocab(self) -> None:
        from whispr.vocab_import import VocabImportDialog
        dialog = VocabImportDialog(self)
        dialog.exec()

    def _flag_restart(self) -> None:
        self._restart_label.setText("Model change requires restart to take effect.")
        self._restart_label.setVisible(True)

    def _on_accept(self) -> None:
        """Collect values, update config, emit signal, close."""
        self._config["model_size"] = self._model_combo.currentText()
        self._config["language"] = self._language_edit.text().strip() or "en"
        self._config["beam_size"] = self._beam_spin.value()
        self._config["injection_method"] = self._inject_combo.currentText()
        self._config["clipboard_threshold_chars"] = self._clipboard_spin.value()
        self._config["audio_device"] = self._device_combo.currentData()
        self._config["hotkey_mode"] = self._mode_combo.currentText()
        self._config["hotkey"] = self._hotkey_edit.text().strip() or "pause"
        self._config["vocabulary_profile"] = self._profile_combo.currentData()

        self._config["postprocessing"]["punctuation_commands"] = self._pp_punctuation.isChecked()
        self._config["postprocessing"]["editing_commands"] = self._pp_editing.isChecked()
        self._config["postprocessing"]["vocabulary_corrections"] = self._pp_corrections.isChecked()
        self._config["postprocessing"]["vocabulary_expansions"] = self._pp_expansions.isChecked()
        self._config["postprocessing"]["auto_capitalisation"] = self._pp_capitalisation.isChecked()
        self._config["postprocessing"]["llm_cleanup"] = self._llm_enabled.isChecked()

        self._config["llm"]["model"] = self._llm_model.text().strip()
        self._config["llm"]["endpoint"] = self._llm_endpoint.text().strip()

        self._config.setdefault("features", {})
        self._config["features"]["audio_feedback"] = self._feat_audio_feedback.isChecked()
        self._config["features"]["preview_overlay"] = self._feat_preview.isChecked()
        self._config["features"]["continuous_mode"] = self._feat_continuous.isChecked()
        self._config["features"]["streaming_transcription"] = self._feat_streaming.isChecked()

        self.settings_changed.emit(self._config)
        self.accept()
