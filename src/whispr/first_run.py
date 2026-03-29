"""First-run setup wizard for Whispr.

Shows a multi-page dialog on first launch when the Whisper model
is not yet downloaded. Guides the user through system checks,
model selection, and model download with progress tracking.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Known model sizes for progress estimation (approximate download bytes)
MODEL_INFO = {
    "medium": {
        "description": "Good accuracy, faster inference (~1.5 GB)",
        "repo": "Systran/faster-whisper-medium",
    },
    "large-v3": {
        "description": "Best accuracy, slower inference (~3.0 GB)",
        "repo": "Systran/faster-whisper-large-v3",
    },
    "large-v3-turbo": {
        "description": "Near-best accuracy, optimised speed (~1.6 GB)",
        "repo": "Systran/faster-whisper-large-v3-turbo",
    },
}


def model_is_cached(model_size: str) -> bool:
    """Check if the faster-whisper model is already downloaded."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    info = MODEL_INFO.get(model_size)
    if not info:
        return False
    repo_dir = "models--" + info["repo"].replace("/", "--")
    snapshot_dir = cache_dir / repo_dir / "snapshots"
    return snapshot_dir.exists() and any(snapshot_dir.iterdir())


class SystemCheckResult:
    """Results of checking system dependencies."""

    def __init__(self) -> None:
        self.xdotool = shutil.which("xdotool") is not None
        self.xclip = shutil.which("xclip") is not None
        self.portaudio = self._check_portaudio()
        self.cuda = self._check_cuda()
        self.issues: list[str] = []

        if not self.xdotool:
            self.issues.append("xdotool not found (sudo apt install xdotool)")
        if not self.xclip:
            self.issues.append("xclip not found (sudo apt install xclip)")
        if not self.portaudio:
            self.issues.append("libportaudio2 not found (sudo apt install libportaudio2)")
        if not self.cuda:
            self.issues.append("CUDA not detected — GPU acceleration unavailable")

    @staticmethod
    def _check_portaudio() -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except (ImportError, Exception):
            return False


class ModelDownloadWorker(QThread):
    """Background worker that downloads the Whisper model."""

    progress = pyqtSignal(int, str)  # percentage, status message
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, model_size: str) -> None:
        super().__init__()
        self.model_size = model_size
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        info = MODEL_INFO.get(self.model_size)
        if not info:
            self.finished.emit(False, f"Unknown model: {self.model_size}")
            return

        self.progress.emit(0, "Starting download...")

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars

            # Disable tqdm bars since we have our own progress
            disable_progress_bars()

            self.progress.emit(10, f"Downloading {self.model_size} from Hugging Face...")

            snapshot_download(
                repo_id=info["repo"],
                allow_patterns=["*"],
            )

            if self._cancelled:
                self.finished.emit(False, "Download cancelled")
                return

            self.progress.emit(100, "Download complete")
            self.finished.emit(True, "Model downloaded successfully")

        except Exception as e:
            logger.error("Model download failed: %s", e)
            self.finished.emit(False, str(e))


class FirstRunWizard(QDialog):
    """Multi-page first-run wizard shown when the Whisper model is not cached."""

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Whispr — First Run Setup")
        self.setMinimumSize(520, 420)
        self._config = config
        self.selected_model = config.get("model_size", "large-v3")
        self._worker: ModelDownloadWorker | None = None

        layout = QVBoxLayout(self)

        # Stacked pages
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_welcome_page())     # 0
        self._pages.addWidget(self._build_model_page())        # 1
        self._pages.addWidget(self._build_download_page())     # 2
        self._pages.addWidget(self._build_complete_page())     # 3
        layout.addWidget(self._pages)

        # Navigation buttons
        nav = QHBoxLayout()
        nav.addStretch()
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        self._back_btn.setVisible(False)
        nav.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        nav.addWidget(self._cancel_btn)
        layout.addLayout(nav)

    # --- Page builders ---

    def _build_welcome_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Welcome to Whispr")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel(
            "Local-only dictation powered by OpenAI Whisper.\n"
            "All processing happens on your device — no data leaves this machine."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # System check
        group = QGroupBox("System Check")
        check_layout = QVBoxLayout(group)

        self._check_result = SystemCheckResult()

        checks = [
            ("xdotool", self._check_result.xdotool, "Text injection"),
            ("xclip", self._check_result.xclip, "Clipboard support"),
            ("libportaudio", self._check_result.portaudio, "Audio capture"),
            ("CUDA GPU", self._check_result.cuda, "GPU acceleration"),
        ]

        for name, ok, desc in checks:
            icon = "\u2705" if ok else "\u274c"
            label = QLabel(f"  {icon}  {name} — {desc}")
            if not ok and name == "CUDA GPU":
                label.setStyleSheet("color: #E65100;")
            elif not ok:
                label.setStyleSheet("color: #C62828;")
            check_layout.addWidget(label)

        layout.addWidget(group)

        if self._check_result.issues:
            issues_label = QLabel(
                "Missing dependencies can be installed with:\n"
                "  sudo apt install xdotool xclip libportaudio2"
            )
            issues_label.setStyleSheet("color: #777; font-size: 11px; margin-top: 8px;")
            layout.addWidget(issues_label)

        layout.addStretch()
        return page

    def _build_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Select Whisper Model")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Choose the speech recognition model to download.\n"
            "Larger models are more accurate but use more GPU memory and disk space."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(12)

        self._model_combo = QComboBox()
        for name, info in MODEL_INFO.items():
            self._model_combo.addItem(f"{name} — {info['description']}", name)

        # Select current config model
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == self.selected_model:
                self._model_combo.setCurrentIndex(i)
                break

        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self._model_combo)

        layout.addSpacing(8)

        self._model_note = QLabel()
        self._model_note.setWordWrap(True)
        self._model_note.setStyleSheet("color: #555; font-size: 12px;")
        self._update_model_note()
        layout.addWidget(self._model_note)

        layout.addStretch()
        return page

    def _build_download_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Downloading Model")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        self._dl_status = QLabel("Preparing download...")
        self._dl_status.setWordWrap(True)
        layout.addWidget(self._dl_status)

        layout.addSpacing(8)

        self._dl_progress = QProgressBar()
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(0)
        layout.addWidget(self._dl_progress)

        self._dl_detail = QLabel("")
        self._dl_detail.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(self._dl_detail)

        layout.addStretch()
        return page

    def _build_complete_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Setup Complete")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        self._complete_msg = QLabel(
            "Whispr is ready to use.\n\n"
            "Hold Pause/Break or F9 to dictate.\n"
            "Right-click the tray icon for settings."
        )
        self._complete_msg.setWordWrap(True)
        layout.addWidget(self._complete_msg)

        layout.addStretch()
        return page

    # --- Navigation ---

    def _go_next(self) -> None:
        current = self._pages.currentIndex()

        if current == 0:
            # Welcome -> Model selection
            self._pages.setCurrentIndex(1)
            self._back_btn.setVisible(True)

        elif current == 1:
            # Model selection -> Download
            self.selected_model = self._model_combo.currentData()

            if model_is_cached(self.selected_model):
                # Skip download, go to complete
                self._pages.setCurrentIndex(3)
                self._next_btn.setText("Start Whispr")
                self._back_btn.setVisible(False)
                self._cancel_btn.setVisible(False)
            else:
                self._pages.setCurrentIndex(2)
                self._next_btn.setEnabled(False)
                self._back_btn.setEnabled(False)
                self._start_download()

        elif current == 2:
            # Download -> Complete (only reachable when download done)
            self._pages.setCurrentIndex(3)
            self._next_btn.setText("Start Whispr")
            self._back_btn.setVisible(False)
            self._cancel_btn.setVisible(False)

        elif current == 3:
            # Complete -> accept
            self.accept()

    def _go_back(self) -> None:
        current = self._pages.currentIndex()
        if current > 0:
            self._pages.setCurrentIndex(current - 1)
            if current - 1 == 0:
                self._back_btn.setVisible(False)

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        self.reject()

    # --- Model selection helpers ---

    def _on_model_changed(self, index: int) -> None:
        self.selected_model = self._model_combo.itemData(index)
        self._update_model_note()

    def _update_model_note(self) -> None:
        if model_is_cached(self.selected_model):
            self._model_note.setText(
                f"\u2705 {self.selected_model} is already downloaded. No download needed."
            )
        else:
            self._model_note.setText(
                f"This model will be downloaded from Hugging Face on the next step.\n"
                f"The download is stored in ~/.cache/huggingface/ and only happens once."
            )

    # --- Download ---

    def _start_download(self) -> None:
        self._dl_status.setText(f"Downloading {self.selected_model}...")
        self._dl_progress.setValue(0)
        self._dl_detail.setText("This may take several minutes depending on your connection.")

        self._worker = ModelDownloadWorker(self.selected_model)
        self._worker.progress.connect(self._on_dl_progress)
        self._worker.finished.connect(self._on_dl_finished)
        self._worker.start()

    def _on_dl_progress(self, pct: int, msg: str) -> None:
        self._dl_progress.setValue(pct)
        self._dl_status.setText(msg)

    def _on_dl_finished(self, success: bool, msg: str) -> None:
        if success:
            self._dl_progress.setValue(100)
            self._dl_status.setText("Download complete!")
            self._dl_detail.setText(msg)
            self._next_btn.setEnabled(True)
            self._next_btn.setText("Next")
            self._back_btn.setEnabled(False)
        else:
            self._dl_status.setText("Download failed")
            self._dl_detail.setText(f"Error: {msg}\n\nYou can retry or select a different model.")
            self._dl_detail.setStyleSheet("color: #C62828; font-size: 11px;")
            self._back_btn.setEnabled(True)
            self._next_btn.setEnabled(False)
