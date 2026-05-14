"""Configurable global hotkey parsing and push-to-talk listener.

Translates a user-configurable hotkey string like ``ctrl+shift+space`` or
``pause`` into a frozenset of pynput key objects, and provides a thin
push-to-talk listener wrapper that fires ``on_combo_active`` once when all
keys are simultaneously held, and ``on_combo_release`` when any of them is
let go.
"""

import logging
from typing import Callable, Optional

try:
    from pynput import keyboard
    _AVAILABLE = True
except ImportError:  # pragma: no cover — pynput is a hard dependency
    keyboard = None
    _AVAILABLE = False

logger = logging.getLogger(__name__)


# Aliases the user might type that don't match pynput.Key attribute names.
_HOTKEY_ALIASES = {
    "control": "ctrl",
    "win": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "option": "alt",
    "opt": "alt",
    "return": "enter",
    "escape": "esc",
    "break": "pause",
    "spacebar": "space",
}


def _build_canon_map() -> dict:
    """Map left/right modifier variants (ctrl_l, ctrl_r, ...) to their generic form."""
    if not _AVAILABLE:
        return {}
    canon = {}
    for name in ("ctrl", "shift", "alt", "cmd"):
        generic = getattr(keyboard.Key, name, None)
        if generic is None:
            continue
        for suffix in ("_l", "_r"):
            variant = getattr(keyboard.Key, name + suffix, None)
            if variant is not None:
                canon[variant] = generic
    return canon


_CANON = _build_canon_map()


def canon_key(key):
    """Return the canonical form of a key (collapsing ctrl_l/ctrl_r → ctrl)."""
    return _CANON.get(key, key)


def parse_hotkey(spec: str) -> Optional[frozenset]:
    """Parse a hotkey spec into a frozenset of pynput key objects.

    Supports modifier names (ctrl/shift/alt/cmd plus aliases above), pynput
    Key attribute names (pause, f9, space, ...), and single ASCII characters.
    Returns None if any segment is unparseable or pynput is unavailable.
    """
    if not _AVAILABLE or not spec:
        return None
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        return None
    keys = []
    for raw in parts:
        name = _HOTKEY_ALIASES.get(raw, raw)
        special = getattr(keyboard.Key, name, None)
        if special is not None:
            keys.append(special)
        elif len(name) == 1:
            keys.append(keyboard.KeyCode.from_char(name))
        else:
            logger.warning("Unparseable hotkey segment: %r", raw)
            return None
    return frozenset(keys)


def start_listener(
    combo: frozenset,
    on_combo_active: Callable[[], None],
    on_combo_release: Callable[[], None],
):
    """Start a global hotkey listener for push-to-talk semantics.

    ``on_combo_active`` fires once when every key in ``combo`` is
    simultaneously held. ``on_combo_release`` fires when any of those keys is
    released after activation. Returns the running listener (or None if
    pynput is unavailable or ``combo`` is empty).
    """
    if not _AVAILABLE or not combo:
        return None

    held: set = set()
    active = False

    def on_press(key):
        nonlocal active
        k = canon_key(key)
        if k not in combo:
            return
        held.add(k)
        if not active and combo.issubset(held):
            active = True
            try:
                on_combo_active()
            except Exception:
                logger.exception("on_combo_active raised")

    def on_release(key):
        nonlocal active
        k = canon_key(key)
        if k not in combo:
            return
        held.discard(k)
        if active:
            active = False
            try:
                on_combo_release()
            except Exception:
                logger.exception("on_combo_release raised")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener
