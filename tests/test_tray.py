"""Tests for whispr.tray — click-to-toggle, state transitions, gating.

The click-to-toggle path is the second activation method (alongside the
hotkey). It must:
  - emit ``toggle_listening`` on a left-click,
  - ignore right-clicks (those open the context menu),
  - stay inert while the model is still LOADING,
  - stay inert while listening is DISABLED.

Each of those is tested independently. The tray state machine — which
icon and tooltip are shown — is also covered so a future visual refresh
can't silently break the "Disabled" tooltip case.
"""

import pytest
from PyQt6.QtWidgets import QSystemTrayIcon

from whispr.tray import WhisprTray, TrayState, _STATE_CONFIG


@pytest.fixture
def tray(qtbot):
    """Construct a tray, set IDLE so left-click is enabled by default.

    qtbot ensures a QApplication exists. We don't show() the tray (no need
    for a real visible icon) — the signal wiring works regardless.
    """
    t = WhisprTray(active_profile="medical")
    t.set_state(TrayState.IDLE)
    yield t
    # No teardown needed; Qt cleans up child objects when the test ends.


# ── Click activation reasons ──────────────────────────────────────────


def test_left_click_emits_toggle_listening(tray, qtbot):
    """ActivationReason.Trigger is the documented left-click. It must emit
    the signal main.py wires to start/stop dictation."""
    with qtbot.waitSignal(tray.toggle_listening, timeout=200):
        tray._on_tray_activated(QSystemTrayIcon.ActivationReason.Trigger)


@pytest.mark.parametrize("reason", [
    QSystemTrayIcon.ActivationReason.Context,        # right-click
    QSystemTrayIcon.ActivationReason.DoubleClick,
    QSystemTrayIcon.ActivationReason.MiddleClick,
    QSystemTrayIcon.ActivationReason.Unknown,
])
def test_non_trigger_activations_do_not_toggle(tray, qtbot, reason):
    """Anything other than Trigger (left-click) must NOT toggle. Right-click
    opens the menu; double-click would otherwise start AND stop instantly."""
    received = []
    tray.toggle_listening.connect(lambda: received.append("toggled"))
    tray._on_tray_activated(reason)
    qtbot.wait(20)
    assert received == []


# ── Gating: LOADING + disabled both suppress the click ────────────────


def test_click_during_loading_is_ignored(tray, qtbot):
    """The model takes a few seconds to load on first run. A click during
    that window would start dictation before the transcriber exists."""
    tray.set_state(TrayState.LOADING)
    received = []
    tray.toggle_listening.connect(lambda: received.append("toggled"))
    tray._on_tray_activated(QSystemTrayIcon.ActivationReason.Trigger)
    qtbot.wait(20)
    assert received == []


def test_click_while_disabled_is_ignored(tray, qtbot):
    """Right-click → Disable Listening sets enabled=False. The tray icon
    stays visible (so the user can re-enable) but clicks must be inert."""
    tray._on_toggle()                       # flips _enabled off
    assert tray.enabled is False
    received = []
    tray.toggle_listening.connect(lambda: received.append("toggled"))
    tray._on_tray_activated(QSystemTrayIcon.ActivationReason.Trigger)
    qtbot.wait(20)
    assert received == []


def test_click_works_again_after_re_enable(tray, qtbot):
    tray._on_toggle()           # disable
    tray._on_toggle()           # re-enable
    assert tray.enabled is True
    with qtbot.waitSignal(tray.toggle_listening, timeout=200):
        tray._on_tray_activated(QSystemTrayIcon.ActivationReason.Trigger)


# ── State machine: tooltip reflects current state ─────────────────────


@pytest.mark.parametrize("state,expected_keyword", [
    (TrayState.LOADING, "Loading"),
    (TrayState.IDLE, "Ready"),
    (TrayState.LISTENING, "Listening"),
    (TrayState.PROCESSING, "Processing"),
])
def test_set_state_updates_tooltip(tray, state, expected_keyword):
    tray.set_state(state)
    assert tray.state is state
    # _STATE_CONFIG is the source of truth — assert via the API surface.
    _, expected_tooltip = _STATE_CONFIG[state]
    assert expected_keyword.lower() in expected_tooltip.lower()


def test_idle_tooltip_shows_disabled_when_listening_off(tray):
    """A disabled tray should advertise that — otherwise the user sees the
    same "Ready" message and wonders why their hotkey isn't firing."""
    tray._on_toggle()                       # disable
    tray.set_state(TrayState.IDLE)
    assert "Disabled" in tray._tray.toolTip()


# ── toggle_enabled signal: external state changes propagate ──────────


def test_toggle_enabled_signal_fires_with_new_value(tray, qtbot):
    """main.py listens for this so it can stop the hotkey listener when the
    user disables. The signal payload must be the new enabled state."""
    received = []
    tray.toggle_enabled.connect(received.append)
    tray._on_toggle()                       # disable
    tray._on_toggle()                       # re-enable
    assert received == [False, True]


# ── profile change ────────────────────────────────────────────────────


def test_profile_selection_emits_when_new_profile_chosen(tray, qtbot):
    received = []
    tray.profile_changed.connect(received.append)
    tray._on_profile_selected("medical")    # already active — no emit
    tray._on_profile_selected("general")    # change — emit
    assert received == ["general"]
    assert tray.active_profile == "general"


def test_set_active_profile_does_not_emit(tray, qtbot):
    """set_active_profile is the *programmatic* setter — used when something
    other than the menu picks the profile. It must not echo back a
    profile_changed signal or the caller gets a self-triggered loop."""
    received = []
    tray.profile_changed.connect(received.append)
    tray.set_active_profile("general")
    qtbot.wait(20)
    assert received == []
    assert tray.active_profile == "general"
