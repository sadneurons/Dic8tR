"""System tray icon and menu for Whispr.

Provides the QSystemTrayIcon with four visual states (loading, idle,
listening, processing), a right-click context menu with vocabulary
profile switching, and Qt signals that other modules connect to.
"""

import logging
from enum import Enum, auto
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction, QActionGroup
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu

from whispr.config import list_profiles

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
        toggle_listening: emitted on left-click — request to start or stop
            a dictation segment (click-to-toggle activation)
        profile_changed(str): emitted when user selects a vocabulary profile
        settings_requested: emitted when user clicks "Settings"
        import_vocab_requested: emitted when user clicks "Import Vocabulary..."
        show_log_requested: emitted when user clicks "Show Last Transcript"
        quit_requested: emitted when user clicks "Quit"
    """

    toggle_enabled = pyqtSignal(bool)
    toggle_listening = pyqtSignal()
    profile_changed = pyqtSignal(str)
    settings_requested = pyqtSignal()
    import_vocab_requested = pyqtSignal()
    show_log_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, active_profile: str = "medical", parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._enabled = True
        self._state = TrayState.LOADING
        self._last_transcript = ""
        self._active_profile = active_profile

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
        self._profile_actions: dict[str, QAction] = {}
        self._build_menu()
        self._tray.setContextMenu(self._menu)

        # Left-click activates dictation (click-to-toggle)
        self._tray.activated.connect(self._on_tray_activated)

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

    @property
    def active_profile(self) -> str:
        return self._active_profile

    def set_active_profile(self, name: str) -> None:
        """Update the checked profile in the menu (called externally)."""
        self._active_profile = name
        action = self._profile_actions.get(name)
        if action:
            action.setChecked(True)

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

        # Vocabulary profile submenu
        profile_menu = QMenu("Vocabulary Profile", self._menu)
        profile_group = QActionGroup(self)
        profile_group.setExclusive(True)

        for name in list_profiles():
            action = QAction(name.capitalize(), self)
            action.setCheckable(True)
            action.setData(name)
            if name == self._active_profile:
                action.setChecked(True)
            action.triggered.connect(lambda checked, n=name: self._on_profile_selected(n))
            profile_group.addAction(action)
            profile_menu.addAction(action)
            self._profile_actions[name] = action

        profile_menu.addSeparator()
        import_action = QAction("Import Vocabulary...", self)
        import_action.triggered.connect(self.import_vocab_requested.emit)
        profile_menu.addAction(import_action)

        self._menu.addMenu(profile_menu)

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

    def _on_profile_selected(self, name: str) -> None:
        if name != self._active_profile:
            self._active_profile = name
            self.profile_changed.emit(name)
            logger.info("Profile selected: %s", name)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Left-click on the tray icon toggles a dictation segment.

        Ignored while loading or when listening is disabled — the user can
        only start/stop dictation once the model is ready.
        """
        if reason != QSystemTrayIcon.ActivationReason.Trigger:
            return
        if not self._enabled or self._state == TrayState.LOADING:
            return
        self.toggle_listening.emit()

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
