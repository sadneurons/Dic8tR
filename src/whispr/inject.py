"""Text injection and editor interaction via xdotool/ydotool.

Handles plain text, special characters (newlines, tabs, bullets),
and editor commands (undo, redo, select, copy, paste, etc).

Special characters are injected via xdotool key rather than xdotool type,
which is unreliable for non-printable characters across applications.
"""

import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

PRE_INJECT_DELAY_S = 0.05

# Map of special characters to xdotool key names
_SPECIAL_KEYS = {
    "\n": "Return",
    "\t": "Tab",
}


def detect_display_server() -> str:
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type in ("x11", "wayland"):
        return session_type
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def _check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _xdotool_key(keys: str) -> bool:
    """Send a key combination via xdotool (e.g. 'ctrl+z', 'Return')."""
    try:
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", keys],
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("xdotool key '%s' failed: %s", keys, e)
        return False


def _xdotool_type_segment(text: str) -> bool:
    """Type a plain text segment (no special chars) via xdotool."""
    if not text:
        return True
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text],
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("xdotool type failed: %s", e)
        return False


def inject_text(
    text: str,
    clipboard_threshold: int = 500,
    method: str = "auto",
    prepend_space: bool = True,
) -> bool:
    """Inject text at the current cursor position.

    Splits text into segments: plain text is typed via xdotool type,
    special characters (newlines, tabs) are sent via xdotool key.
    This ensures consistent behaviour across all applications.
    """
    if not text:
        logger.warning("inject_text() called with empty text")
        return False

    if prepend_space and text[0] not in ("\n",):
        text = " " + text

    if method == "auto":
        method = detect_display_server()

    time.sleep(PRE_INJECT_DELAY_S)

    if method == "x11":
        return _inject_x11_segmented(text, clipboard_threshold)
    elif method == "wayland":
        return _inject_wayland(text, clipboard_threshold)
    else:
        logger.error("Unknown display server: %s", method)
        return False


def _inject_x11_segmented(text: str, clipboard_threshold: int) -> bool:
    """Inject text on X11, splitting special characters into key events."""
    if not _check_tool("xdotool"):
        logger.error("xdotool not found. Install with: sudo apt install xdotool")
        return False

    # For very long text, use clipboard paste (faster)
    if len(text) > clipboard_threshold:
        return _clipboard_paste_x11(text)

    # Split into segments of plain text and special characters
    segments = _split_segments(text)

    for seg_type, seg_value in segments:
        if seg_type == "text":
            if not _xdotool_type_segment(seg_value):
                return False
        elif seg_type == "key":
            if not _xdotool_key(seg_value):
                return False

    logger.info("Injected %d chars via segmented xdotool", len(text))
    return True


def _split_segments(text: str) -> list[tuple[str, str]]:
    """Split text into ('text', 'plain string') and ('key', 'keyname') segments."""
    segments = []
    buf = []

    for ch in text:
        if ch in _SPECIAL_KEYS:
            if buf:
                segments.append(("text", "".join(buf)))
                buf = []
            segments.append(("key", _SPECIAL_KEYS[ch]))
        else:
            buf.append(ch)

    if buf:
        segments.append(("text", "".join(buf)))

    return segments


def _clipboard_paste_x11(text: str) -> bool:
    """Copy text to clipboard via xclip, then Ctrl+V via xdotool."""
    if not _check_tool("xclip"):
        logger.error("xclip not found. Install with: sudo apt install xclip")
        return False

    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
            timeout=5,
        )
        time.sleep(0.02)
        _xdotool_key("ctrl+v")
        logger.info("Injected %d chars via clipboard paste (X11)", len(text))
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Clipboard paste (X11) failed: %s", e)
        return False


# ── Editor interaction commands ───────────────────────────────────────

def send_undo(count: int = 1, method: str = "auto") -> bool:
    """Send Ctrl+Z to the focused window."""
    if method == "auto":
        method = detect_display_server()
    time.sleep(PRE_INJECT_DELAY_S)
    for _ in range(count):
        if method == "x11":
            if not _xdotool_key("ctrl+z"):
                return False
        elif method == "wayland":
            _ydotool_key_combo("ctrl+z")
    logger.info("Sent undo x%d", count)
    return True


def send_redo(method: str = "auto") -> bool:
    """Send Ctrl+Shift+Z to the focused window."""
    if method == "auto":
        method = detect_display_server()
    time.sleep(PRE_INJECT_DELAY_S)
    if method == "x11":
        return _xdotool_key("ctrl+shift+z")
    return False


def send_key_combo(combo: str, method: str = "auto") -> bool:
    """Send an arbitrary key combination (e.g. 'ctrl+s', 'ctrl+b').

    Used by the custom voice command system.
    """
    if method == "auto":
        method = detect_display_server()
    time.sleep(PRE_INJECT_DELAY_S)
    if method == "x11":
        return _xdotool_key(combo)
    elif method == "wayland":
        return _ydotool_key_combo(combo)
    return False


def send_backspace(count: int = 1, method: str = "auto") -> bool:
    """Send Backspace key presses."""
    if method == "auto":
        method = detect_display_server()
    time.sleep(PRE_INJECT_DELAY_S)
    for _ in range(count):
        if method == "x11":
            if not _xdotool_key("BackSpace"):
                return False
    return True


def send_delete_word(method: str = "auto") -> bool:
    """Send Ctrl+Backspace to delete the previous word."""
    if method == "auto":
        method = detect_display_server()
    time.sleep(PRE_INJECT_DELAY_S)
    if method == "x11":
        return _xdotool_key("ctrl+BackSpace")
    return False


# ── Wayland support ───────────────────────────────────────────────────

def _inject_wayland(text: str, clipboard_threshold: int) -> bool:
    if len(text) > clipboard_threshold:
        return _clipboard_paste_wayland(text)
    return _ydotool_type(text)


def _ydotool_type(text: str) -> bool:
    if not _check_tool("ydotool"):
        logger.error("ydotool not found")
        return False
    try:
        subprocess.run(
            ["ydotool", "type", "--key-delay", "0", "--", text],
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("ydotool type failed: %s", e)
        return False


def _ydotool_key_combo(combo: str) -> bool:
    """Send a key combo via ydotool. Translates 'ctrl+z' to ydotool keycodes."""
    # ydotool uses raw evdev keycodes — this is a simplified mapping
    _KEYMAP = {
        "ctrl": "29", "shift": "42", "alt": "56", "super": "125",
        "a": "30", "b": "48", "c": "46", "i": "23", "s": "31",
        "v": "47", "x": "45", "z": "52",
    }
    if not _check_tool("ydotool"):
        return False

    parts = combo.lower().split("+")
    codes = []
    for p in parts:
        code = _KEYMAP.get(p)
        if code is None:
            logger.warning("Unknown ydotool key: %s", p)
            return False
        codes.append(code)

    # Press all, release all in reverse
    args = []
    for c in codes:
        args.extend([f"{c}:1"])
    for c in reversed(codes):
        args.extend([f"{c}:0"])

    try:
        subprocess.run(["ydotool", "key"] + args, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("ydotool key combo failed: %s", e)
        return False


def _clipboard_paste_wayland(text: str) -> bool:
    if not _check_tool("wl-copy") or not _check_tool("ydotool"):
        return False
    try:
        subprocess.run(["wl-copy", "--", text], check=True, timeout=5)
        time.sleep(0.02)
        _ydotool_key_combo("ctrl+v")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Clipboard paste (Wayland) failed: %s", e)
        return False
