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
