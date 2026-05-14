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
    """Detect which display server to drive for input injection.

    Prefers x11 whenever DISPLAY is set, even on a Wayland session — under
    XWayland the app we're injecting into is an X11 client, and xdotool works
    while ydotool would require root for /dev/uinput. Only commit to wayland
    when there is genuinely no X path available.
    """
    if os.environ.get("DISPLAY"):
        return "x11"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type in ("x11", "wayland"):
        return session_type
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


# ── Clipboard save / restore ──────────────────────────────────────────
#
# Privacy note: when we route a transcript through the system clipboard,
# we save the previous contents, paste, then restore them. This prevents
# the dictated text from lingering for the next app that pastes — but it
# does NOT prevent active clipboard managers (Klipper, GPaste, CopyQ,
# Parcellite) from snapshotting the dictated text the moment we set the
# clipboard. To fully avoid that, dictation should bypass the clipboard
# path entirely (force xdotool/ydotool type). That's a follow-up.

_CLIPBOARD_RESTORE_DELAY_S = 0.1


def _read_clipboard_x11() -> bytes | None:
    """Return current X11 CLIPBOARD selection contents, or None on failure."""
    if not _check_tool("xclip"):
        return None
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True,
            timeout=1,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def _write_clipboard_x11(data: bytes) -> None:
    """Set the X11 CLIPBOARD selection. Empty data clears it."""
    if not _check_tool("xclip"):
        return
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=data,
            timeout=1,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Failed to restore X11 clipboard: %s", e)


def _read_clipboard_wayland() -> bytes | None:
    """Return current Wayland clipboard contents, or None on failure."""
    if not _check_tool("wl-paste"):
        return None
    try:
        result = subprocess.run(
            ["wl-paste", "-n"],
            capture_output=True,
            timeout=1,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def _write_clipboard_wayland(data: bytes) -> None:
    """Set the Wayland clipboard. Empty data clears it."""
    if not _check_tool("wl-copy"):
        return
    try:
        if not data:
            subprocess.run(["wl-copy", "--clear"], timeout=1)
        else:
            subprocess.run(["wl-copy", "--"], input=data, timeout=1)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Failed to restore Wayland clipboard: %s", e)


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
    """Copy text to clipboard via xclip, then Ctrl+V via xdotool.

    Saves the prior clipboard contents and restores them after paste so the
    dictated text does not linger in the clipboard for the next app to read.
    """
    if not _check_tool("xclip"):
        logger.error("xclip not found. Install with: sudo apt install xclip")
        return False

    saved = _read_clipboard_x11()
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
            timeout=5,
        )
        time.sleep(0.02)
        # _xdotool_key catches its own subprocess errors and returns False
        # rather than raising; honor that so callers know whether the paste
        # actually landed (the finally below still runs either way).
        if not _xdotool_key("ctrl+v"):
            logger.error("Clipboard paste (X11) failed: ctrl+v keystroke did not send")
            return False
        logger.info("Injected %d chars via clipboard paste (X11)", len(text))
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Clipboard paste (X11) failed: %s", e)
        return False
    finally:
        # Let the target app finish absorbing the paste before we overwrite
        # the clipboard, otherwise some apps re-read mid-paste and get empty.
        time.sleep(_CLIPBOARD_RESTORE_DELAY_S)
        _write_clipboard_x11(saved if saved is not None else b"")


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
    """Send N Backspace key presses.

    Uses xdotool's --repeat so a 300-char delete is one subprocess invocation
    instead of 300. The --delay 0 keeps it fast; some apps drop events at
    full speed, so 5ms is a tolerable upper bound.
    """
    if count <= 0:
        return True
    if method == "auto":
        method = detect_display_server()
    time.sleep(PRE_INJECT_DELAY_S)
    if method == "x11":
        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers",
                 "--repeat", str(count), "--delay", "5", "BackSpace"],
                check=True,
                timeout=max(5, count // 50),
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error("xdotool BackSpace x%d failed: %s", count, e)
            return False
    if method == "wayland":
        # Wayland path falls back to per-press; ydotool doesn't have --repeat
        for _ in range(count):
            if not _ydotool_key_combo("backspace"):
                return False
        return True
    return False


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
    """Send a key combo via ydotool. Translates names to evdev keycodes.

    The keymap below mirrors every action command in postprocess.ACTION_COMMANDS
    plus the common modifier+letter combos. Names are matched case-insensitively
    (callers split on '+' and lower-case the parts). Add entries here when new
    action commands are introduced — silently dropping a combo means the
    Wayland injection path no-ops without telling the user.
    """
    # ydotool uses raw evdev keycodes — see /usr/include/linux/input-event-codes.h
    _KEYMAP = {
        # modifiers
        "ctrl": "29", "shift": "42", "alt": "56", "super": "125",
        # letters
        "a": "30", "b": "48", "c": "46", "d": "32", "e": "18", "f": "33",
        "g": "34", "h": "35", "i": "23", "j": "36", "k": "37", "l": "38",
        "m": "50", "n": "49", "o": "24", "p": "25", "q": "16", "r": "19",
        "s": "31", "t": "20", "u": "22", "v": "47", "w": "17", "x": "45",
        "y": "21", "z": "44",
        # editing / navigation — needed by action commands
        "home": "102", "end": "107",
        "prior": "104", "pageup": "104",
        "next": "109", "pagedown": "109",
        "backspace": "14", "delete": "111",
        "tab": "15", "enter": "28", "return": "28", "escape": "1", "esc": "1",
        "left": "105", "right": "106", "up": "103", "down": "108",
        # function keys
        "f1": "59", "f2": "60", "f3": "61", "f4": "62", "f5": "63",
        "f6": "64", "f7": "65", "f8": "66", "f9": "67", "f10": "68",
        "f11": "87", "f12": "88",
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
    """Copy text to clipboard via wl-copy, then Ctrl+V via ydotool.

    Saves the prior clipboard contents and restores them after paste so the
    dictated text does not linger in the clipboard for the next app to read.
    """
    if not _check_tool("wl-copy") or not _check_tool("ydotool"):
        return False
    saved = _read_clipboard_wayland()
    try:
        subprocess.run(["wl-copy", "--", text], check=True, timeout=5)
        time.sleep(0.02)
        # _ydotool_key_combo swallows its subprocess errors and returns False;
        # propagate that to the caller so a failed paste doesn't masquerade as
        # success (the finally below still restores the clipboard either way).
        if not _ydotool_key_combo("ctrl+v"):
            logger.error("Clipboard paste (Wayland) failed: ctrl+v keystroke did not send")
            return False
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Clipboard paste (Wayland) failed: %s", e)
        return False
    finally:
        time.sleep(_CLIPBOARD_RESTORE_DELAY_S)
        _write_clipboard_wayland(saved if saved is not None else b"")
