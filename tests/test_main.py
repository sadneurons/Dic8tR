"""Tests for WhisprApp action handlers.

The scratch-that handler has to undo whatever the last injection actually
did. Clipboard paste is a single atomic edit → one Ctrl+Z reverses it.
xdotool type writes char-by-char; most editors record each character as a
separate undo step, so one Ctrl+Z would only drop a single letter — we
have to send a Backspace burst of the exact length instead.

These tests verify the right helper fires for each injection method.
"""

from unittest.mock import patch

import pytest

from whispr.main import WhisprApp


def _stub_app(*, last_text: str = "", clipboard: bool = False,
              chars: int = 0) -> WhisprApp:
    """Build a WhisprApp with just the fields the action handler reads.

    Skips __init__ entirely (which constructs the tray, audio capture,
    transcriber, preview overlay, LLM client, recording lock, and a stack
    of Qt signal wiring) — _handle_action only needs self.config plus the
    three _last_injection_* attributes.
    """
    app = WhisprApp.__new__(WhisprApp)
    app.config = {"injection_method": "auto"}
    app._last_text = last_text
    app._last_injection_clipboard = clipboard
    app._last_injection_chars = chars
    return app


@patch("whispr.main.send_backspace")
@patch("whispr.main.send_undo")
def test_scratch_that_after_clipboard_paste_uses_ctrl_z(
    mock_undo, mock_backspace,
) -> None:
    """Clipboard paste is one atomic edit in the target editor's undo
    history. One Ctrl+Z reverses it cleanly — no need to backspace."""
    app = _stub_app(last_text="long clipboard-pasted text",
                    clipboard=True, chars=300)
    app._handle_action("ACTION:SCRATCH_THAT")

    mock_undo.assert_called_once()
    mock_backspace.assert_not_called()


@patch("whispr.main.send_backspace")
@patch("whispr.main.send_undo")
def test_scratch_that_after_xdotool_type_uses_backspace_burst(
    mock_undo, mock_backspace,
) -> None:
    """xdotool type writes char-by-char; in most editors each char is its
    own undo step. So we backspace the exact number of characters that
    were actually sent (including the leading space inject_text prepends).
    """
    app = _stub_app(last_text="hello world", clipboard=False, chars=12)
    app._handle_action("ACTION:SCRATCH_THAT")

    mock_backspace.assert_called_once()
    args, _kwargs = mock_backspace.call_args
    assert args[0] == 12, (
        f"expected backspace count = 12 (recorded injection chars), got {args[0]}"
    )
    mock_undo.assert_not_called()


@patch("whispr.main.send_backspace")
@patch("whispr.main.send_undo")
def test_scratch_that_with_no_prior_injection_is_a_noop(
    mock_undo, mock_backspace,
) -> None:
    """If nothing has been injected yet, scratch-that should do nothing
    rather than firing a stray Ctrl+Z or backspace into whatever happens
    to be focused."""
    app = _stub_app(last_text="", clipboard=False, chars=0)
    app._handle_action("ACTION:SCRATCH_THAT")

    mock_undo.assert_not_called()
    mock_backspace.assert_not_called()


@patch("whispr.main.send_undo")
def test_undo_action_dispatches_to_send_undo(mock_undo) -> None:
    """Plain UNDO (distinct from SCRATCH_THAT) always sends Ctrl+Z, no
    matter what the last injection was — it's a request to undo whatever
    is in the editor right now."""
    app = _stub_app(last_text="something", clipboard=False, chars=10)
    app._handle_action("ACTION:UNDO")
    mock_undo.assert_called_once()
