"""Tests for the configurable-hotkey parser."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pynput import keyboard

from whispr.hotkey import parse_hotkey, canon_key


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
