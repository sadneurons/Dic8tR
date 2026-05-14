"""Tests for whispr.inject — clipboard save/restore and display detection.

The save → set → paste → restore wrapper is a privacy mitigation: without
it, dictated text lingers in the clipboard for whatever app pastes next.
These tests pin down the call order (save before paste, restore after) and
the two failure modes that must still leave a clean clipboard.

Display-server detection is also exercised because item G's XWayland fix
(prefer x11 when DISPLAY is set) is a behaviour change that's easy to
regress on.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from whispr.inject import (
    _clipboard_paste_x11,
    detect_display_server,
)


# ── Clipboard save / restore ──────────────────────────────────────────


def _make_run(read_response: bytes | None = b"PRIOR",
              paste_raises: bool = False):
    """Build a subprocess.run side_effect that fakes xclip and xdotool.

    `read_response` is what `xclip -o` returns; None means returncode 1
    (read failed). `paste_raises` makes the xdotool key call raise
    CalledProcessError, simulating a paste failure that must still trigger
    the finally-block restore.
    """
    def _run(args, *, input=None, capture_output=False, timeout=None,
             check=False, **kwargs):
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.stderr = b""
        if args and args[0] == "xclip":
            if "-o" in args:
                # Read path
                if read_response is None:
                    result.returncode = 1
                    result.stdout = b""
                else:
                    result.returncode = 0
                    result.stdout = read_response
                return result
            # Write path — succeeds silently
            result.returncode = 0
            result.stdout = b""
            return result
        if args and args[0] == "xdotool":
            if paste_raises:
                raise subprocess.CalledProcessError(1, args)
            result.returncode = 0
            result.stdout = b""
            return result
        # Anything else we don't care about
        result.returncode = 0
        result.stdout = b""
        return result
    return _run


@patch("whispr.inject._check_tool", return_value=True)
@patch("whispr.inject.subprocess.run")
def test_clipboard_paste_saves_then_restores_prior(mock_run, _check) -> None:
    """Happy path: prior clipboard had content; paste must restore it."""
    saved = b"PRIOR CONTENTS"
    mock_run.side_effect = _make_run(read_response=saved)

    ok = _clipboard_paste_x11("dictated text")
    assert ok is True

    cmds = [c.args[0] for c in mock_run.call_args_list]
    inputs = [c.kwargs.get("input") for c in mock_run.call_args_list]

    # Order: read → write new → ctrl+v → write back
    assert cmds[0] == ["xclip", "-selection", "clipboard", "-o"]
    assert cmds[1] == ["xclip", "-selection", "clipboard"]
    assert inputs[1] == b"dictated text"
    assert cmds[2][:2] == ["xdotool", "key"]
    assert "ctrl+v" in cmds[2]
    assert cmds[3] == ["xclip", "-selection", "clipboard"]
    assert inputs[3] == saved


@patch("whispr.inject._check_tool", return_value=True)
@patch("whispr.inject.subprocess.run")
def test_clipboard_paste_clears_when_prior_unreadable(mock_run, _check) -> None:
    """If we can't read the prior clipboard, restore must write empty bytes
    (clearing it) rather than leaving the dictated text behind."""
    mock_run.side_effect = _make_run(read_response=None)

    ok = _clipboard_paste_x11("dictated text")
    assert ok is True

    inputs = [c.kwargs.get("input") for c in mock_run.call_args_list]
    # Last call is the restore — empty since the read failed
    assert inputs[-1] == b""


@patch("whispr.inject._check_tool", return_value=True)
@patch("whispr.inject.subprocess.run")
def test_clipboard_paste_restores_even_when_paste_fails(mock_run, _check) -> None:
    """The restore lives in a finally block. If the xdotool ctrl+v fails, the
    prior clipboard must still come back — otherwise our dictated text is
    left in the clipboard, defeating the whole point of save/restore."""
    saved = b"PRIOR"
    mock_run.side_effect = _make_run(read_response=saved, paste_raises=True)

    ok = _clipboard_paste_x11("dictated text")
    assert ok is False

    inputs = [c.kwargs.get("input") for c in mock_run.call_args_list]
    assert inputs[-1] == saved


# ── Display-server detection (item G) ─────────────────────────────────


def test_detect_display_server_prefers_x11_when_DISPLAY_set(monkeypatch) -> None:
    """XWayland sessions have both DISPLAY and WAYLAND_DISPLAY set. We pick
    x11 because xdotool works against XWayland clients, whereas ydotool
    requires uinput access we typically don't have."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    assert detect_display_server() == "x11"


def test_detect_display_server_returns_wayland_when_only_wayland_set(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    assert detect_display_server() == "wayland"


def test_detect_display_server_falls_back_to_session_type(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    assert detect_display_server() == "x11"


def test_detect_display_server_unknown_when_nothing_set(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    assert detect_display_server() == "unknown"


# ── Module-level senders (send_undo / send_redo / send_key_combo /
#    send_delete_word / send_backspace) ──────────────────────────────


from whispr.inject import (  # noqa: E402 — grouped with the senders block
    send_undo,
    send_redo,
    send_key_combo,
    send_delete_word,
    send_backspace,
    _split_segments,
)


@pytest.fixture
def fake_run(monkeypatch):
    """Recorder for subprocess.run that fakes xdotool / ydotool success.

    Exposes ``.calls`` as the list of argv lists in order, so tests can
    assert which binary was invoked with which flags. Use the ``set_fail``
    closure to make a future call raise CalledProcessError, simulating a
    failed keystroke.
    """
    calls = []
    state = {"fail_next": False}

    class _Recorder:
        @staticmethod
        def __call__(args, *_, **__):
            calls.append(list(args))
            if state["fail_next"]:
                state["fail_next"] = False
                raise subprocess.CalledProcessError(1, args)
            return subprocess.CompletedProcess(args=args, returncode=0,
                                               stdout=b"", stderr=b"")

        @staticmethod
        def set_fail():
            state["fail_next"] = True

    rec = _Recorder()
    monkeypatch.setattr("whispr.inject.subprocess.run", rec.__call__)
    monkeypatch.setattr("whispr.inject._check_tool", lambda _: True)
    # Skip the 50ms pre-inject pause so tests stay fast.
    monkeypatch.setattr("whispr.inject.PRE_INJECT_DELAY_S", 0)
    rec.calls = calls
    return rec


def test_send_undo_x11_invokes_xdotool_ctrl_z(fake_run) -> None:
    assert send_undo(method="x11") is True
    assert fake_run.calls == [
        ["xdotool", "key", "--clearmodifiers", "ctrl+z"],
    ]


def test_send_undo_count_repeats(fake_run) -> None:
    """A count > 1 must send N separate Ctrl+Z events. Some editors only
    undo one atomic operation per keystroke, so the count matters."""
    send_undo(count=3, method="x11")
    assert len(fake_run.calls) == 3
    for call in fake_run.calls:
        assert call[-1] == "ctrl+z"


def test_send_redo_x11(fake_run) -> None:
    assert send_redo(method="x11") is True
    assert fake_run.calls[-1][-1] == "ctrl+shift+z"


def test_send_delete_word_x11(fake_run) -> None:
    assert send_delete_word(method="x11") is True
    assert fake_run.calls[-1][-1] == "ctrl+BackSpace"


def test_send_key_combo_passes_through_to_xdotool(fake_run) -> None:
    assert send_key_combo("ctrl+a", method="x11") is True
    assert fake_run.calls[-1][-1] == "ctrl+a"


def test_send_backspace_uses_repeat_flag(fake_run) -> None:
    """send_backspace should issue ONE subprocess call with --repeat N,
    not N separate calls. The latter pegged a CPU core on long deletes."""
    send_backspace(count=120, method="x11")
    assert len(fake_run.calls) == 1
    cmd = fake_run.calls[0]
    assert "--repeat" in cmd
    assert cmd[cmd.index("--repeat") + 1] == "120"
    assert cmd[-1] == "BackSpace"


def test_send_backspace_with_zero_count_is_noop(fake_run) -> None:
    """Zero-count is the natural value of _last_injection_chars after a
    state clear. It must NOT pass --repeat 0 to xdotool, which is a noisy
    error rather than a no-op."""
    assert send_backspace(count=0, method="x11") is True
    assert fake_run.calls == []


def test_send_backspace_negative_count_is_noop(fake_run) -> None:
    assert send_backspace(count=-5, method="x11") is True
    assert fake_run.calls == []


def test_send_undo_returns_false_on_xdotool_failure(fake_run) -> None:
    fake_run.set_fail()
    assert send_undo(method="x11") is False


def test_send_redo_unsupported_on_wayland(fake_run) -> None:
    """The ydotool path doesn't implement redo (no ctrl+shift+z mapping yet);
    the function returns False rather than silently no-opping with True."""
    assert send_redo(method="wayland") is False


def test_senders_return_false_for_unknown_method(fake_run) -> None:
    """A garbled config value (e.g. 'X11' instead of 'x11') should be a
    visible failure, not a silent no-op. detect_display_server is only used
    when method='auto'."""
    assert send_key_combo("ctrl+s", method="unknown") is False
    assert send_delete_word(method="unknown") is False
    assert send_backspace(count=5, method="unknown") is False


# ── _split_segments: text vs special-key partitioning ────────────────


def test_split_segments_plain_text_is_one_segment() -> None:
    assert _split_segments("hello world") == [("text", "hello world")]


def test_split_segments_newline_becomes_key() -> None:
    """\\n must come out as a Return keypress, not as a literal '\\n' typed —
    xdotool type can't reliably produce a true newline in every app."""
    segs = _split_segments("a\nb")
    assert segs == [("text", "a"), ("key", "Return"), ("text", "b")]


def test_split_segments_tab_becomes_key() -> None:
    segs = _split_segments("col1\tcol2")
    assert segs == [("text", "col1"), ("key", "Tab"), ("text", "col2")]


def test_split_segments_leading_special() -> None:
    segs = _split_segments("\nstart")
    assert segs == [("key", "Return"), ("text", "start")]


def test_split_segments_consecutive_specials() -> None:
    segs = _split_segments("a\n\nb")
    assert segs == [("text", "a"), ("key", "Return"), ("key", "Return"), ("text", "b")]


def test_split_segments_empty_input() -> None:
    assert _split_segments("") == []
