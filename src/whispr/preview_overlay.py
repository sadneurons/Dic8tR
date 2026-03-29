"""Live preview overlay for Whispr.

A small floating window that shows transcription results before injection.
The user can accept (Enter/click), edit, or dismiss (Esc) the text.
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class PreviewOverlay(QWidget):
    """Floating overlay showing transcription before injection.

    Signals:
        accepted(str): user accepted the text (possibly edited)
        dismissed(): user cancelled the injection
    """

    accepted = pyqtSignal(str)
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Whispr Preview")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.setMinimumWidth(500)
        self.setMaximumHeight(200)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                border: 2px solid #89b4fa;
                border-radius: 8px;
            }
            QTextEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: none;
                font-size: 14px;
                padding: 4px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton#dismiss {
                background-color: #585b70;
                color: #cdd6f4;
            }
            QLabel {
                color: #6c7086;
                font-size: 10px;
                border: none;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        # Editable text area
        self._text_edit = QTextEdit()
        self._text_edit.setAcceptRichText(False)
        self._text_edit.setFont(QFont("monospace", 13))
        self._text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._text_edit)

        # Button row
        btn_row = QHBoxLayout()
        hint = QLabel("Enter = inject  |  Esc = discard  |  Edit text above")
        btn_row.addWidget(hint)
        btn_row.addStretch()

        dismiss_btn = QPushButton("Discard")
        dismiss_btn.setObjectName("dismiss")
        dismiss_btn.clicked.connect(self._on_dismiss)
        btn_row.addWidget(dismiss_btn)

        accept_btn = QPushButton("Inject")
        accept_btn.clicked.connect(self._on_accept)
        btn_row.addWidget(accept_btn)
        layout.addLayout(btn_row)

    def show_text(self, text: str) -> None:
        """Show the overlay with the given text, ready for editing."""
        self._text_edit.setPlainText(text)
        self._text_edit.selectAll()

        # Position at bottom-center of screen
        screen = self.screen()
        if screen:
            geo = screen.availableGeometry()
            self.adjustSize()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + geo.height() - self.height() - 60
            self.move(x, y)

        self.show()
        self.raise_()
        self.activateWindow()
        self._text_edit.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._on_dismiss()
        elif event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self._on_accept()
        else:
            super().keyPressEvent(event)

    def _on_accept(self) -> None:
        text = self._text_edit.toPlainText().strip()
        self.hide()
        if text:
            self.accepted.emit(text)
        else:
            self.dismissed.emit()

    def _on_dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()
