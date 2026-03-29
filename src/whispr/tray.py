"""System tray icon and menu for Whispr.

Provides the QSystemTrayIcon with four visual states (loading, idle,
listening, processing), a right-click context menu, and Qt signals
that other modules connect to.
"""

import logging
from enum import Enum, auto
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu

logger = logging.getLogger(__name__)

ICON_DIR = Path(__file__).parent / "icons"


class TrayState(Enum):
    LOADING = auto()     # model loading at startup
    IDLE = auto()        # ready, waiting for hotkey
    LISTENING = auto()   # hotkey held, recording
    PROCESSING = auto()  # transcribing / post-processing


# Map states to icon files and tooltip text
_STATE_CONFIG = {
    TrayState.LOADING:    ("mic_loading.svg",    "Whispr — Loading model..."),
    TrayState.IDLE:       ("mic_idle.svg",       "Whispr — Ready"),
    TrayState.LISTENING:  ("mic_listening.svg",  "Whispr — Listening..."),
    TrayState.PROCESSING: ("mic_processing.svg", "Whispr — Processing..."),
}


class WhisprTray(QObject):
    """System tray icon manager.

    Signals:
        toggle_enabled: emitted when user toggles listening on/off
        settings_requested: emitted when user clicks "Settings"
        show_log_requested: emitted when user clicks "Show Last Transcript"
        quit_requested: emitted when user clicks "Quit"
    """

    toggle_enabled = pyqtSignal(bool)
    settings_requested = pyqtSignal()
    show_log_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._enabled = True
        self._state = TrayState.LOADING
        self._last_transcript = ""

        # Load icons
        self._icons: dict[TrayState, QIcon] = {}
        for state, (filename, _) in _STATE_CONFIG.items():
            icon_path = ICON_DIR / filename
            if icon_path.exists():
                self._icons[state] = QIcon(str(icon_path))
            else:
                logger.warning("Icon not found: %s", icon_path)
                self._icons[state] = QIcon()

        # Create tray icon
        self._tray = QSystemTrayIcon(self._icons[TrayState.LOADING])
        self._tray.setToolTip("Whispr — Loading model...")

        # Build context menu
        self._menu = QMenu()
        self._build_menu()
        self._tray.setContextMenu(self._menu)

    def show(self) -> None:
        """Show the tray icon."""
        self._tray.show()
        logger.info("Tray icon visible")

    def hide(self) -> None:
        """Hide the tray icon."""
        self._tray.hide()

    def set_state(self, state: TrayState) -> None:
        """Update the tray icon and tooltip to reflect the current state."""
        self._state = state
        icon = self._icons.get(state)
        if icon:
            self._tray.setIcon(icon)

        _, tooltip = _STATE_CONFIG[state]
        if not self._enabled and state == TrayState.IDLE:
            tooltip = "Whispr — Disabled"
        self._tray.setToolTip(tooltip)

        logger.debug("Tray state: %s", state.name)

    @property
    def state(self) -> TrayState:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_last_transcript(self, text: str) -> None:
        """Store the last transcript for the log viewer."""
        self._last_transcript = text

    def show_notification(self, title: str, message: str, duration_ms: int = 3000) -> None:
        """Show a system notification from the tray."""
        self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, duration_ms)

    def _build_menu(self) -> None:
        """Build the right-click context menu."""
        # Toggle listening
        self._toggle_action = QAction("Disable Listening", self)
        self._toggle_action.setCheckable(False)
        self._toggle_action.triggered.connect(self._on_toggle)
        self._menu.addAction(self._toggle_action)

        self._menu.addSeparator()

        # Show last transcript
        show_log_action = QAction("Show Last Transcript", self)
        show_log_action.triggered.connect(self._on_show_log)
        self._menu.addAction(show_log_action)

        # Settings
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self.settings_requested.emit)
        self._menu.addAction(settings_action)

        self._menu.addSeparator()

        # Quit
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_requested.emit)
        self._menu.addAction(quit_action)

    def _on_toggle(self) -> None:
        self._enabled = not self._enabled
        self._toggle_action.setText(
            "Disable Listening" if self._enabled else "Enable Listening"
        )
        if not self._enabled:
            self.set_state(TrayState.IDLE)
        self.toggle_enabled.emit(self._enabled)
        logger.info("Listening %s", "enabled" if self._enabled else "disabled")

    def _on_show_log(self) -> None:
        if self._last_transcript:
            self.show_notification("Last Transcript", self._last_transcript, 5000)
        else:
            self.show_notification("Whispr", "No transcript yet.", 2000)
        self.show_log_requested.emit()
