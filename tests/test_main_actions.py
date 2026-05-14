"""Tests for WhisprApp._handle_action — specifically the scratch-that branch.

scratch-that has to choose the right reversal for the last injection: a
single Ctrl+Z when the text went through the clipboard (atomic paste), or
a backspace burst of exactly the injected length when the text was typed
character-by-character via xdotool (because most editors then record each
character as its own undo step and a single Ctrl+Z would only drop one
letter).

We construct WhisprApp via ``__new__`` so we can populate the few fields
the handler reads without dragging in the Qt tray, the audio device, the
Whisper model, or the LLM client. The handler only touches:

  - ``self.config["injection_method"]``
  - ``self._last_text``
  - ``self._last_injection_clipboard``
  - ``self._last_injection_chars``

and calls module-level helpers (``send_undo``, ``send_backspace``,
``send_delete_word``, ``send_redo``, ``send_key_combo``) that we replace
with recorders.
"""

import pytest

from whispr import main as main_module
from whispr.main import WhisprApp


@pytest.fixture
def app(monkeypatch):
    """A bare WhisprApp instance with just the fields _handle_action reads.

    Replaces inject helpers with recorders so the test can assert exactly
    which one was called and with what arguments.
    """
    instance = WhisprApp.__new__(WhisprApp)
    instance.config = {"injection_method": "x11"}
    instance._last_text = ""
    instance._last_injection_chars = 0
    instance._last_injection_clipboard = False

    calls: list[tuple[str, tuple, dict]] = []

    def _recorder(name):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return True
        return _fn

    monkeypatch.setattr(main_module, "send_undo", _recorder("send_undo"))
    monkeypatch.setattr(main_module, "send_backspace", _recorder("send_backspace"))
    monkeypatch.setattr(main_module, "send_redo", _recorder("send_redo"))
    monkeypatch.setattr(main_module, "send_delete_word", _recorder("send_delete_word"))
    monkeypatch.setattr(main_module, "send_key_combo", _recorder("send_key_combo"))
    monkeypatch.setattr(main_module, "inject_text", lambda *a, **k: True)

    instance._calls = calls  # type: ignore[attr-defined]
    return instance


# ── SCRATCH_THAT: three branches ──────────────────────────────────────


def test_scratch_that_with_no_prior_text_does_nothing(app):
    """No prior injection → handler must not send any key event. A naive
    Ctrl+Z would otherwise undo whatever the user did manually."""
    app._last_text = ""

    app._handle_action("ACTION:SCRATCH_THAT")

    assert app._calls == [], f"expected no calls, got {app._calls}"


def test_scratch_that_after_clipboard_paste_sends_one_undo(app):
    """Clipboard paste is one atomic edit, so a single Ctrl+Z reverses it.
    Sending N backspaces would over-delete and eat the user's prior text."""
    app._last_text = "the patient was confused"
    app._last_injection_clipboard = True
    app._last_injection_chars = 25  # leading space + 24 chars (informational)

    app._handle_action("ACTION:SCRATCH_THAT")

    names = [c[0] for c in app._calls]
    assert names == ["send_undo"]
    # Must use the configured injection method, not "auto" or hard-coded x11
    assert app._calls[0][2].get("method") == "x11"


def test_scratch_that_after_typed_inject_sends_exact_backspace_count(app):
    """xdotool type writes character-by-character; most editors track each as
    a separate undo step, so a single Ctrl+Z only drops one letter. Backspace
    the exact prepended length instead."""
    app._last_text = "hello world"
    app._last_injection_clipboard = False
    app._last_injection_chars = 12   # leading space + 11 chars

    app._handle_action("ACTION:SCRATCH_THAT")

    names = [c[0] for c in app._calls]
    assert names == ["send_backspace"]
    # Positional or keyword — accept either to keep the test robust to a
    # future refactor of the call site.
    args, kwargs = app._calls[0][1], app._calls[0][2]
    count = args[0] if args else kwargs.get("count")
    assert count == 12
    assert kwargs.get("method") == "x11"


def test_scratch_that_clears_state_so_second_invocation_is_noop(app):
    """After one scratch-that the tracked injection must be wiped, otherwise
    a second 'scratch that' would over-undo into the user's prior content."""
    app._last_text = "first dictation"
    app._last_injection_clipboard = False
    app._last_injection_chars = 16

    app._handle_action("ACTION:SCRATCH_THAT")
    app._handle_action("ACTION:SCRATCH_THAT")

    # Only the first invocation should have sent a backspace burst.
    names = [c[0] for c in app._calls]
    assert names == ["send_backspace"]
    # And the state is now zeroed.
    assert app._last_text == ""
    assert app._last_injection_chars == 0
    assert app._last_injection_clipboard is False


def test_scratch_that_uses_configured_method(app):
    """Wayland users should get the Wayland code path, not the x11 default."""
    app.config["injection_method"] = "wayland"
    app._last_text = "x"
    app._last_injection_clipboard = True
    app._last_injection_chars = 2

    app._handle_action("ACTION:SCRATCH_THAT")

    assert app._calls[0][2].get("method") == "wayland"


# ── Sanity: other action branches still call the right helpers ────────


def test_undo_action_calls_send_undo(app):
    app._handle_action("ACTION:UNDO")
    assert [c[0] for c in app._calls] == ["send_undo"]


def test_redo_action_calls_send_redo(app):
    app._handle_action("ACTION:REDO")
    assert [c[0] for c in app._calls] == ["send_redo"]


def test_scratch_word_action_calls_send_delete_word(app):
    app._handle_action("ACTION:SCRATCH_WORD")
    assert [c[0] for c in app._calls] == ["send_delete_word"]


def test_key_action_dispatches_per_comma_separated_combo(app):
    """Compound actions like 'delete line' are encoded as a comma list
    (Home,shift+End,BackSpace). Each combo must go through send_key_combo."""
    app._handle_action("ACTION:KEY:Home,shift+End,BackSpace")
    names = [c[0] for c in app._calls]
    assert names == ["send_key_combo", "send_key_combo", "send_key_combo"]
    combos = [c[1][0] for c in app._calls]
    assert combos == ["Home", "shift+End", "BackSpace"]
