"""
tests/test_os_mouse.py
======================
Unit tests for the OsMouse driver in eye_mouse/os_mouse.py.

All Win32 calls (SendInput, GetSystemMetrics) are replaced by MagicMock
objects so these tests run safely on any platform, including CI runners
without real hardware.

Patching strategy
-----------------
We patch ``os_mouse.ctypes.windll.user32`` at the module level so that
OsMouse.__init__ and every _send / _send_batch call use the mock rather
than the real user32.dll.  sys.platform is patched to 'win32' for most
tests; a dedicated test verifies the RuntimeError on non-Windows.
"""

import sys
import os
import threading
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path setup — mirror conftest.py's approach
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

import os_mouse as _os_mouse_module  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user32_mock(screen_w: int = 1920, screen_h: int = 1080) -> MagicMock:
    """Return a pre-configured user32 mock with sane screen dimensions."""
    mock = MagicMock()
    mock.GetSystemMetrics.side_effect = lambda metric: screen_w if metric == 0 else screen_h
    mock.SendInput.return_value = 1  # SUCCESS for single-event calls
    return mock


def _make_mouse(user32_mock: MagicMock):
    """
    Instantiate OsMouse with sys.platform forced to 'win32' and
    ctypes.windll.user32 replaced by *user32_mock*.
    """
    with patch('os_mouse.ctypes.windll') as mock_windll, \
         patch('os_mouse.sys') as mock_sys:
        mock_sys.platform = 'win32'
        mock_windll.user32 = user32_mock
        from os_mouse import OsMouse
        instance = OsMouse()
    return instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def user32():
    """Fresh user32 mock for each test."""
    return _make_user32_mock()


@pytest.fixture()
def mouse(user32):
    """OsMouse instance wired to *user32* mock."""
    return _make_mouse(user32)


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestOsMouseInit:
    """Tests for OsMouse.__init__ and screen-metric reading."""

    def test_initialization(self):
        """OsMouse reads screen_w and screen_h via GetSystemMetrics(0/1)."""
        user32 = _make_user32_mock(screen_w=2560, screen_h=1440)
        m = _make_mouse(user32)

        assert m.screen_size == (2560, 1440), (
            "screen_size must reflect values returned by GetSystemMetrics"
        )
        calls_made = [c.args[0] for c in user32.GetSystemMetrics.call_args_list]
        assert 0 in calls_made, "GetSystemMetrics(0) / SM_CXSCREEN not called"
        assert 1 in calls_made, "GetSystemMetrics(1) / SM_CYSCREEN not called"

    def test_non_windows_raises(self):
        """OsMouse.__init__ raises RuntimeError when sys.platform != 'win32'."""
        with patch('os_mouse.sys') as mock_sys:
            mock_sys.platform = 'linux'
            from os_mouse import OsMouse
            with pytest.raises(RuntimeError, match="OsMouse requer Windows"):
                OsMouse()


# ---------------------------------------------------------------------------
# Tests: move()
# ---------------------------------------------------------------------------

class TestOsMouseMove:
    """Tests for OsMouse.move()."""

    def test_move_calls_sendinput(self, mouse, user32):
        """move(x, y) calls SendInput with MOUSEEVENTF_MOVE|MOUSEEVENTF_ABSOLUTE flags."""
        mouse.move(100, 200)

        assert user32.SendInput.called, "SendInput must be called by move()"
        n_inputs = user32.SendInput.call_args.args[0]
        assert n_inputs == 1, "move() must send exactly 1 INPUT event"

    def test_move_blocked_by_emergency_stop(self, mouse, user32):
        """move() silently returns without calling SendInput after emergency_stop()."""
        mouse.emergency_stop()
        user32.SendInput.reset_mock()

        mouse.move(500, 500)

        user32.SendInput.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: left_click, right_click, double_click
# ---------------------------------------------------------------------------

class TestOsMouseClicks:
    """Tests for left_click, right_click, double_click."""

    def test_left_click_atomic(self, mouse, user32):
        """left_click() calls SendInput once with 2 events (LEFTDOWN + LEFTUP)."""
        user32.SendInput.return_value = 2

        mouse.left_click()

        assert user32.SendInput.call_count == 1, "left_click must use a single batch SendInput"
        n_inputs = user32.SendInput.call_args.args[0]
        assert n_inputs == 2, "Batch must contain exactly 2 events: LEFTDOWN and LEFTUP"

    def test_right_click_atomic(self, mouse, user32):
        """right_click() calls SendInput once with 2 events (RIGHTDOWN + RIGHTUP)."""
        user32.SendInput.return_value = 2

        mouse.right_click()

        assert user32.SendInput.call_count == 1
        n_inputs = user32.SendInput.call_args.args[0]
        assert n_inputs == 2, "right_click batch must contain RIGHTDOWN + RIGHTUP"

    def test_double_click_atomic(self, mouse, user32):
        """double_click() calls SendInput once with exactly 4 events."""
        user32.SendInput.return_value = 4

        mouse.double_click()

        assert user32.SendInput.call_count == 1
        n_inputs = user32.SendInput.call_args.args[0]
        assert n_inputs == 4, "double_click batch must contain 4 events"

    def test_clicks_blocked_by_emergency_stop(self, mouse, user32):
        """No click events are sent after emergency_stop()."""
        mouse.emergency_stop()
        user32.SendInput.reset_mock()

        mouse.left_click()
        mouse.right_click()
        mouse.double_click()

        user32.SendInput.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: button_down / button_up
# ---------------------------------------------------------------------------

class TestOsMouseButtonDownUp:
    """Tests for button_down() and button_up()."""

    def test_button_down_left(self, mouse, user32):
        """button_down('left') sends LEFTDOWN; a second call is a noop (idempotent)."""
        mouse.button_down('left')
        assert user32.SendInput.call_count == 1

        user32.SendInput.reset_mock()
        mouse.button_down('left')  # already down — must be noop
        user32.SendInput.assert_not_called()

    def test_button_up_left(self, mouse, user32):
        """button_up('left') sends LEFTUP only when _left_down is True."""
        # Not pressed -> should be noop
        mouse.button_up('left')
        user32.SendInput.assert_not_called()

        # Press then release
        mouse.button_down('left')
        user32.SendInput.reset_mock()
        mouse.button_up('left')
        assert user32.SendInput.call_count == 1

    def test_button_up_skips_when_not_pressed(self, mouse, user32):
        """button_up with no prior button_down must not call SendInput."""
        mouse.button_up('right')
        user32.SendInput.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: scroll()
# ---------------------------------------------------------------------------

class TestOsMouseScroll:
    """Tests for scroll()."""

    def test_scroll(self, mouse, user32):
        """scroll(120) sends MOUSEEVENTF_WHEEL with mouseData=120 as one event."""
        mouse.scroll(120)

        assert user32.SendInput.called
        n_inputs = user32.SendInput.call_args.args[0]
        assert n_inputs == 1, "scroll must send a single INPUT event"

    def test_scroll_blocked_by_emergency_stop(self, mouse, user32):
        """scroll() is blocked after emergency_stop()."""
        mouse.emergency_stop()
        user32.SendInput.reset_mock()

        mouse.scroll(-120)

        user32.SendInput.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: release_all()
# ---------------------------------------------------------------------------

class TestOsMouseReleaseAll:
    """Tests for release_all() idempotency."""

    def test_release_all_idempotent(self, mouse, user32):
        """release_all() called 3 times with no button pressed: SendInput never called.
        Then button_down('left') + release_all() twice: only one LEFTUP sent."""
        # Part 1: nothing pressed
        mouse.release_all()
        mouse.release_all()
        mouse.release_all()
        user32.SendInput.assert_not_called()

        # Part 2: press then release twice
        mouse.button_down('left')
        user32.SendInput.reset_mock()

        mouse.release_all()
        assert user32.SendInput.call_count == 1, "First release_all must send LEFTUP"

        user32.SendInput.reset_mock()
        mouse.release_all()
        user32.SendInput.assert_not_called()

    def test_release_all_not_blocked_by_emergency_stop(self, mouse, user32):
        """release_all() must execute regardless of emergency_stopped state."""
        mouse.button_down('left')
        # Force emergency state without going through emergency_stop() side-effects
        mouse._emergency_stopped = True
        user32.SendInput.reset_mock()

        mouse.release_all()

        assert user32.SendInput.call_count == 1, (
            "release_all() must send LEFTUP even when emergency_stopped is True"
        )


# ---------------------------------------------------------------------------
# Tests: emergency_stop() / reset_emergency_stop()
# ---------------------------------------------------------------------------

class TestOsMouseEmergencyStop:
    """Tests for emergency_stop() and reset_emergency_stop()."""

    def test_emergency_stop_blocks_events(self, mouse, user32):
        """After emergency_stop(), move/click/scroll all send nothing."""
        mouse.emergency_stop()
        user32.SendInput.reset_mock()

        mouse.move(10, 10)
        mouse.left_click()
        mouse.right_click()
        mouse.double_click()
        mouse.scroll(120)

        user32.SendInput.assert_not_called()

    def test_reset_emergency_stop(self, mouse, user32):
        """After reset_emergency_stop(), events work again."""
        mouse.emergency_stop()
        mouse.reset_emergency_stop()
        user32.SendInput.reset_mock()

        mouse.move(300, 400)
        mouse.left_click()

        assert user32.SendInput.call_count >= 2, (
            "Events must be sent again after reset_emergency_stop()"
        )

    def test_is_emergency_stopped_property(self, mouse):
        """is_emergency_stopped reflects the current emergency state."""
        assert mouse.is_emergency_stopped is False
        mouse.emergency_stop()
        assert mouse.is_emergency_stopped is True
        mouse.reset_emergency_stop()
        assert mouse.is_emergency_stopped is False


# ---------------------------------------------------------------------------
# Tests: button_state property
# ---------------------------------------------------------------------------

class TestOsMouseButtonState:
    """Tests for the button_state dict property."""

    def test_button_state_tracking(self, mouse, user32):
        """button_state reflects the internal pressed state accurately."""
        assert mouse.button_state == {'left_down': False, 'right_down': False}

        mouse.button_down('left')
        assert mouse.button_state['left_down'] is True
        assert mouse.button_state['right_down'] is False

        mouse.button_down('right')
        assert mouse.button_state['left_down'] is True
        assert mouse.button_state['right_down'] is True

        mouse.button_up('left')
        assert mouse.button_state['left_down'] is False
        assert mouse.button_state['right_down'] is True

        mouse.release_all()
        assert mouse.button_state == {'left_down': False, 'right_down': False}
