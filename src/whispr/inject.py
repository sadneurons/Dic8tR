"""Text injection at cursor position via xdotool/ydotool.

Detects the display server (X11 or Wayland) and uses the appropriate
tool to type text into the currently focused window. Falls back to
clipboard-paste for long strings to avoid slow keystroke injection.
"""

import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

# Delay after hotkey release before injecting, to let modifier keys settle
PRE_INJECT_DELAY_S = 0.05


def detect_display_server() -> str:
    """Detect the current display server.

    Returns "x11", "wayland", or "unknown".
    """
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type in ("x11", "wayland"):
        return session_type

    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"

    if os.environ.get("DISPLAY"):
        return "x11"

    return "unknown"


def _check_tool(name: str) -> bool:
    """Check if a command-line tool is available on PATH."""
    return shutil.which(name) is not None


def inject_text(
    text: str,
    clipboard_threshold: int = 500,
    method: str = "auto",
    prepend_space: bool = True,
) -> bool:
    """Inject text at the current cursor position.

    Args:
        text: The text to inject.
        clipboard_threshold: If text exceeds this length, use clipboard paste
            instead of keystroke injection for speed.
        method: "auto" (detect display server), "x11", or "wayland".
        prepend_space: If True, add a leading space so consecutive utterances
            don't run together. Skipped if text starts with newline.

    Returns:
        True if injection succeeded, False otherwise.
    """
    if not text:
        logger.warning("inject_text() called with empty text")
        return False

    if prepend_space and not text[0] in ("\n",):
        text = " " + text

    if method == "auto":
        method = detect_display_server()

    time.sleep(PRE_INJECT_DELAY_S)

    if method == "x11":
        return _inject_x11(text, clipboard_threshold)
    elif method == "wayland":
        return _inject_wayland(text, clipboard_threshold)
    else:
        logger.error("Unknown display server: %s", method)
        return False


def _inject_x11(text: str, clipboard_threshold: int) -> bool:
    """Inject text on X11 using xdotool, with xclip clipboard fallback."""
    use_clipboard = len(text) > clipboard_threshold

    if use_clipboard:
        return _clipboard_paste_x11(text)
    else:
        return _xdotool_type(text)


def _xdotool_type(text: str) -> bool:
    """Type text using xdotool keystroke injection."""
    if not _check_tool("xdotool"):
        logger.error("xdotool not found. Install with: sudo apt install xdotool")
        return False

    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text],
            check=True,
            timeout=10,
        )
        logger.info("Injected %d chars via xdotool type", len(text))
        return True
    except subprocess.CalledProcessError as e:
        logger.error("xdotool type failed: %s", e)
        return False
    except subprocess.TimeoutExpired:
        logger.error("xdotool type timed out")
        return False


def _clipboard_paste_x11(text: str) -> bool:
    """Copy text to clipboard via xclip, then simulate Ctrl+V via xdotool."""
    if not _check_tool("xclip"):
        logger.error("xclip not found. Install with: sudo apt install xclip")
        return False
    if not _check_tool("xdotool"):
        logger.error("xdotool not found. Install with: sudo apt install xdotool")
        return False

    try:
        # Copy to clipboard
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
            timeout=5,
        )
        # Small delay for clipboard to settle
        time.sleep(0.02)
        # Paste
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            check=True,
            timeout=5,
        )
        logger.info("Injected %d chars via clipboard paste (X11)", len(text))
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Clipboard paste (X11) failed: %s", e)
        return False


def _inject_wayland(text: str, clipboard_threshold: int) -> bool:
    """Inject text on Wayland using ydotool, with wl-copy clipboard fallback."""
    use_clipboard = len(text) > clipboard_threshold

    if use_clipboard:
        return _clipboard_paste_wayland(text)
    else:
        return _ydotool_type(text)


def _ydotool_type(text: str) -> bool:
    """Type text using ydotool keystroke injection."""
    if not _check_tool("ydotool"):
        logger.error(
            "ydotool not found. Install with: sudo apt install ydotool "
            "(requires ydotoold daemon running)"
        )
        return False

    try:
        subprocess.run(
            ["ydotool", "type", "--key-delay", "0", "--", text],
            check=True,
            timeout=10,
        )
        logger.info("Injected %d chars via ydotool type", len(text))
        return True
    except subprocess.CalledProcessError as e:
        logger.error("ydotool type failed: %s", e)
        return False
    except subprocess.TimeoutExpired:
        logger.error("ydotool type timed out")
        return False


def _clipboard_paste_wayland(text: str) -> bool:
    """Copy text to clipboard via wl-copy, then simulate Ctrl+V via ydotool."""
    if not _check_tool("wl-copy"):
        logger.error("wl-copy not found. Install with: sudo apt install wl-clipboard")
        return False
    if not _check_tool("ydotool"):
        logger.error("ydotool not found. Install with: sudo apt install ydotool")
        return False

    try:
        subprocess.run(
            ["wl-copy", "--", text],
            check=True,
            timeout=5,
        )
        time.sleep(0.02)
        subprocess.run(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],  # Ctrl+V keycodes
            check=True,
            timeout=5,
        )
        logger.info("Injected %d chars via clipboard paste (Wayland)", len(text))
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Clipboard paste (Wayland) failed: %s", e)
        return False
