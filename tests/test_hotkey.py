"""Tests for the configurable-hotkey parser and push-to-talk listener."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pynput import keyboard

from whispr import hotkey as hotkey_mod
from whispr.hotkey import parse_hotkey, canon_key, start_listener


class TestParseHotkey:
    def test_single_special_key(self):
        assert parse_hotkey("pause") == frozenset({keyboard.Key.pause})

    def test_function_key(self):
        assert parse_hotkey("f9") == frozenset({keyboard.Key.f9})

    def test_combo(self):
        result = parse_hotkey("ctrl+shift+space")
        assert result == frozenset({
            keyboard.Key.ctrl,
            keyboard.Key.shift,
            keyboard.Key.space,
        })

    def test_modifier_plus_letter(self):
        result = parse_hotkey("alt+x")
        assert keyboard.Key.alt in result
        # The 'x' becomes a KeyCode; compare via from_char.
        assert keyboard.KeyCode.from_char("x") in result

    def test_alias_control(self):
        assert parse_hotkey("control+space") == parse_hotkey("ctrl+space")

    def test_alias_super_is_cmd(self):
        assert parse_hotkey("super") == frozenset({keyboard.Key.cmd})

    def test_alias_break_is_pause(self):
        assert parse_hotkey("break") == frozenset({keyboard.Key.pause})

    def test_case_insensitive(self):
        assert parse_hotkey("CTRL+Shift+Space") == parse_hotkey("ctrl+shift+space")

    def test_strips_empty_segments(self):
        # 'ctrl+' has a trailing empty segment that should be ignored.
        assert parse_hotkey("ctrl+") == frozenset({keyboard.Key.ctrl})

    def test_empty_returns_none(self):
        assert parse_hotkey("") is None
        assert parse_hotkey("+") is None

    def test_unparseable_returns_none(self):
        # Multi-char segment that isn't a Key attribute and isn't an alias.
        assert parse_hotkey("not_a_real_key") is None
        assert parse_hotkey("ctrl+definitelynotakey") is None


class TestCanonKey:
    def test_left_ctrl_collapses_to_ctrl(self):
        assert canon_key(keyboard.Key.ctrl_l) == keyboard.Key.ctrl

    def test_right_shift_collapses_to_shift(self):
        assert canon_key(keyboard.Key.shift_r) == keyboard.Key.shift

    def test_unknown_key_passes_through(self):
        # KeyCode-style keys are not in the canonical map; pass through.
        c = keyboard.KeyCode.from_char("c")
        assert canon_key(c) == c


class _FakeListener:
    """Stand-in for pynput.keyboard.Listener used in start_listener tests.

    The real listener spawns a background thread that grabs keyboard
    events from the OS. We don't want that during unit tests. By replacing
    the class with this fake, start_listener still constructs and "starts"
    a listener, but we capture the on_press/on_release callbacks so we
    can drive them directly with synthetic key events.
    """

    instances: list["_FakeListener"] = []

    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = False
        self.started = False
        self.stopped = False
        _FakeListener.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_listener(monkeypatch):
    _FakeListener.instances.clear()
    monkeypatch.setattr(hotkey_mod.keyboard, "Listener", _FakeListener)
    yield _FakeListener
    _FakeListener.instances.clear()


class TestStartListenerSingleKey:
    """Push-to-talk semantics for a single trigger key (the default Pause)."""

    def test_press_fires_active_callback(self, fake_listener):
        combo = parse_hotkey("pause")
        active, release = [], []
        listener = start_listener(combo, lambda: active.append(1),
                                  lambda: release.append(1))
        assert listener is not None
        assert listener.started is True

        listener.on_press(keyboard.Key.pause)
        assert active == [1]
        assert release == []

    def test_release_after_press_fires_release(self, fake_listener):
        combo = parse_hotkey("pause")
        active, release = [], []
        listener = start_listener(combo, lambda: active.append(1),
                                  lambda: release.append(1))

        listener.on_press(keyboard.Key.pause)
        listener.on_release(keyboard.Key.pause)
        assert release == [1]

    def test_release_without_press_is_noop(self, fake_listener):
        """If the user releases a key they never pressed (the combo wasn't
        active), the release callback must not fire — otherwise stop_recording
        would be called when nothing was recording."""
        combo = parse_hotkey("pause")
        release = []
        listener = start_listener(combo, lambda: None, lambda: release.append(1))
        listener.on_release(keyboard.Key.pause)
        assert release == []

    def test_unrelated_key_does_not_fire(self, fake_listener):
        """Pressing a key NOT in the configured combo must not trigger."""
        combo = parse_hotkey("pause")
        active = []
        listener = start_listener(combo, lambda: active.append(1), lambda: None)
        listener.on_press(keyboard.Key.f9)
        assert active == []


class TestStartListenerChord:
    """Multi-key chords (ctrl+shift+space) — every key must be held
    simultaneously, and releasing any one of them ends the active state."""

    def test_all_keys_required_for_active(self, fake_listener):
        combo = parse_hotkey("ctrl+shift+space")
        active = []
        listener = start_listener(combo, lambda: active.append(1), lambda: None)

        listener.on_press(keyboard.Key.ctrl)
        assert active == []          # one of three — not yet
        listener.on_press(keyboard.Key.shift)
        assert active == []          # two of three — not yet
        listener.on_press(keyboard.Key.space)
        assert active == [1]         # full chord — fire once

    def test_repeated_press_does_not_double_fire(self, fake_listener):
        """Holding a key generates repeat key events on most OSes. The
        listener must dedupe them so on_combo_active fires exactly once."""
        combo = parse_hotkey("pause")
        active = []
        listener = start_listener(combo, lambda: active.append(1), lambda: None)

        listener.on_press(keyboard.Key.pause)
        listener.on_press(keyboard.Key.pause)
        listener.on_press(keyboard.Key.pause)
        assert active == [1]

    def test_releasing_any_chord_key_fires_release_once(self, fake_listener):
        combo = parse_hotkey("ctrl+shift+space")
        release = []
        listener = start_listener(combo, lambda: None,
                                  lambda: release.append(1))

        listener.on_press(keyboard.Key.ctrl)
        listener.on_press(keyboard.Key.shift)
        listener.on_press(keyboard.Key.space)
        listener.on_release(keyboard.Key.shift)   # release one — counts
        assert release == [1]

        # Subsequent releases of the remaining keys must not re-fire.
        listener.on_release(keyboard.Key.ctrl)
        listener.on_release(keyboard.Key.space)
        assert release == [1]

    def test_left_modifier_variants_count_as_canonical(self, fake_listener):
        """A real keyboard sends ctrl_l, ctrl_r etc. The listener must treat
        them as their canonical form so 'ctrl+x' fires whether the user
        holds left or right Ctrl."""
        combo = parse_hotkey("ctrl+x")
        active = []
        listener = start_listener(combo, lambda: active.append(1), lambda: None)

        listener.on_press(keyboard.Key.ctrl_l)
        listener.on_press(keyboard.KeyCode.from_char("x"))
        assert active == [1]


class TestStartListenerGuards:
    def test_empty_combo_returns_none(self, fake_listener):
        """An unparseable hotkey from config produces an empty/None combo.
        start_listener must refuse rather than spawn a listener that fires
        on every keystroke."""
        assert start_listener(frozenset(), lambda: None, lambda: None) is None

    def test_callbacks_that_raise_are_logged_not_propagated(self, fake_listener,
                                                            caplog):
        """A buggy on_combo_active must not crash the listener thread — it
        would silently break dictation for the rest of the session."""
        combo = parse_hotkey("pause")
        def boom(): raise RuntimeError("kaboom")
        listener = start_listener(combo, boom, boom)

        listener.on_press(keyboard.Key.pause)   # must not raise
        listener.on_release(keyboard.Key.pause) # must not raise
        # Both exceptions were logged.
        assert any("on_combo_active" in r.getMessage() for r in caplog.records)
        assert any("on_combo_release" in r.getMessage() for r in caplog.records)
