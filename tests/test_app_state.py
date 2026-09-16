"""
tests/test_app_state.py
=======================
Unit tests for StateMachine and AppState in eye_mouse/app_state.py.

No external mocking is required — the module is pure Python.
Tests cover state transitions, invalid-transition errors, release_all
callback semantics, listener notification, thread safety, and the
valid_transitions() diagnostic helper.
"""

import sys
import os
import threading
from unittest.mock import MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from app_state import AppState, StateMachine, InvalidTransition  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sm(initial: AppState = AppState.INITIALIZING,
        release_all=None) -> StateMachine:
    """Convenience factory for StateMachine."""
    return StateMachine(initial=initial, on_release_all=release_all)


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------

class TestStateMachineInit:
    """Tests covering StateMachine initialisation."""

    def test_initial_state(self):
        """StateMachine starts in INITIALIZING by default."""
        sm = _sm()
        assert sm.state is AppState.INITIALIZING

    def test_custom_initial_state(self):
        """StateMachine honours an explicit initial state argument."""
        sm = _sm(initial=AppState.CALIBRATING)
        assert sm.state is AppState.CALIBRATING


# ---------------------------------------------------------------------------
# Tests: transitions
# ---------------------------------------------------------------------------

class TestStateMachineTransitions:
    """Tests covering valid and invalid transitions."""

    def test_valid_transition(self):
        """INITIALIZING -> CALIBRATING is a valid transition."""
        sm = _sm()
        sm.transition(AppState.CALIBRATING)
        assert sm.state is AppState.CALIBRATING

    def test_invalid_transition_raises(self):
        """CALIBRATING -> PAUSED raises InvalidTransition with correct from/to states."""
        sm = _sm(initial=AppState.CALIBRATING)
        with pytest.raises(InvalidTransition) as exc_info:
            sm.transition(AppState.PAUSED)
        err = exc_info.value
        assert err.from_state is AppState.CALIBRATING
        assert err.to_state is AppState.PAUSED
        assert "CALIBRATING" in str(err)
        assert "PAUSED" in str(err)

    def test_same_state_is_noop(self):
        """Transitioning to the current state is silently ignored."""
        cb = MagicMock()
        sm = _sm(initial=AppState.INITIALIZING, release_all=cb)

        sm.transition(AppState.INITIALIZING)  # same state

        cb.assert_not_called()
        assert sm.state is AppState.INITIALIZING

    def test_full_happy_path(self):
        """INITIALIZING -> CALIBRATING -> ACTIVE -> PAUSED -> ACTIVE -> SHUTTING_DOWN."""
        sm = _sm()
        path = [
            AppState.CALIBRATING,
            AppState.ACTIVE,
            AppState.PAUSED,
            AppState.ACTIVE,
            AppState.SHUTTING_DOWN,
        ]
        for state in path:
            sm.transition(state)
        assert sm.state is AppState.SHUTTING_DOWN

    def test_shutting_down_is_terminal(self):
        """No transitions are allowed from SHUTTING_DOWN."""
        sm = _sm(initial=AppState.SHUTTING_DOWN)
        with pytest.raises(InvalidTransition):
            sm.transition(AppState.ACTIVE)
        assert sm.state is AppState.SHUTTING_DOWN


# ---------------------------------------------------------------------------
# Tests: mouse_allowed
# ---------------------------------------------------------------------------

class TestMouseAllowed:
    """Tests for the mouse_allowed property."""

    @pytest.mark.parametrize("state,expected", [
        (AppState.INITIALIZING,  False),
        (AppState.CALIBRATING,   False),
        (AppState.ACTIVE,        True),
        (AppState.PAUSED,        False),
        (AppState.TRACKING_LOST, False),
        (AppState.ERROR,         False),
        (AppState.SHUTTING_DOWN, False),
    ])
    def test_mouse_allowed_only_in_active(self, state, expected):
        """mouse_allowed is True only when state is ACTIVE."""
        sm = _sm(initial=state)
        assert sm.mouse_allowed is expected, (
            f"mouse_allowed should be {expected} in state {state.name}"
        )


# ---------------------------------------------------------------------------
# Tests: release_all callback
# ---------------------------------------------------------------------------

class TestReleaseAllCallback:
    """Tests verifying when on_release_all is called during transitions."""

    def test_release_all_called_on_pause(self):
        """ACTIVE -> PAUSED calls on_release_all."""
        cb = MagicMock()
        sm = _sm(initial=AppState.ACTIVE, release_all=cb)
        sm.transition(AppState.PAUSED)
        cb.assert_called_once()

    def test_release_all_called_on_shutting_down(self):
        """Any valid path to SHUTTING_DOWN triggers on_release_all."""
        cb = MagicMock()
        sm = _sm(initial=AppState.ACTIVE, release_all=cb)
        sm.transition(AppState.SHUTTING_DOWN)
        cb.assert_called_once()

    def test_release_all_called_on_tracking_lost(self):
        """ACTIVE -> TRACKING_LOST calls on_release_all."""
        cb = MagicMock()
        sm = _sm(initial=AppState.ACTIVE, release_all=cb)
        sm.transition(AppState.TRACKING_LOST)
        cb.assert_called_once()

    def test_release_all_called_on_calibrating(self):
        """ACTIVE -> CALIBRATING calls on_release_all."""
        cb = MagicMock()
        sm = _sm(initial=AppState.ACTIVE, release_all=cb)
        sm.transition(AppState.CALIBRATING)
        cb.assert_called_once()

    def test_release_all_not_called_on_active(self):
        """PAUSED -> ACTIVE does NOT call on_release_all (ACTIVE is not in RELEASE_ON_ENTER)."""
        cb = MagicMock()
        sm = _sm(initial=AppState.PAUSED, release_all=cb)
        sm.transition(AppState.ACTIVE)
        cb.assert_not_called()

    def test_release_all_not_called_when_no_callback(self):
        """StateMachine works fine with on_release_all=None."""
        sm = _sm(initial=AppState.ACTIVE, release_all=None)
        sm.transition(AppState.PAUSED)  # should not raise
        assert sm.state is AppState.PAUSED


# ---------------------------------------------------------------------------
# Tests: listeners
# ---------------------------------------------------------------------------

class TestListeners:
    """Tests for add_listener / listener notification."""

    def test_listener_called_on_transition(self):
        """Listener receives (from_state, to_state) args on each transition."""
        sm = _sm()
        received = []
        sm.add_listener(lambda f, t: received.append((f, t)))

        sm.transition(AppState.CALIBRATING)
        sm.transition(AppState.ACTIVE)

        assert received == [
            (AppState.INITIALIZING, AppState.CALIBRATING),
            (AppState.CALIBRATING,  AppState.ACTIVE),
        ]

    def test_listener_exception_does_not_propagate(self):
        """A listener that raises must not bubble up to the caller."""
        sm = _sm()

        def bad_listener(f, t):
            raise RuntimeError("listener error")

        sm.add_listener(bad_listener)
        sm.transition(AppState.CALIBRATING)  # must not raise
        assert sm.state is AppState.CALIBRATING

    def test_listener_not_called_on_same_state(self):
        """Listeners are not called when transitioning to the same state (noop)."""
        sm = _sm()
        calls = []
        sm.add_listener(lambda f, t: calls.append((f, t)))

        sm.transition(AppState.INITIALIZING)  # same state

        assert calls == []


# ---------------------------------------------------------------------------
# Tests: try_transition
# ---------------------------------------------------------------------------

class TestTryTransition:
    """Tests for the try_transition helper."""

    def test_try_transition_returns_true_on_valid(self):
        """try_transition returns True for a valid transition."""
        sm = _sm()
        result = sm.try_transition(AppState.CALIBRATING)
        assert result is True
        assert sm.state is AppState.CALIBRATING

    def test_try_transition_returns_false_on_invalid(self):
        """try_transition returns False for an invalid transition, state unchanged."""
        sm = _sm(initial=AppState.CALIBRATING)
        result = sm.try_transition(AppState.PAUSED)
        assert result is False
        assert sm.state is AppState.CALIBRATING


# ---------------------------------------------------------------------------
# Tests: valid_transitions property
# ---------------------------------------------------------------------------

class TestValidTransitions:
    """Tests for the valid_transitions() diagnostic method."""

    @pytest.mark.parametrize("state,expected_targets", [
        (AppState.INITIALIZING,  {AppState.CALIBRATING, AppState.ERROR, AppState.SHUTTING_DOWN}),
        (AppState.CALIBRATING,   {AppState.ACTIVE, AppState.ERROR, AppState.SHUTTING_DOWN}),
        (AppState.ACTIVE,        {AppState.PAUSED, AppState.CALIBRATING, AppState.TRACKING_LOST,
                                  AppState.ERROR, AppState.SHUTTING_DOWN}),
        (AppState.PAUSED,        {AppState.ACTIVE, AppState.CALIBRATING, AppState.SHUTTING_DOWN}),
        (AppState.TRACKING_LOST, {AppState.ACTIVE, AppState.PAUSED, AppState.ERROR,
                                  AppState.SHUTTING_DOWN}),
        (AppState.ERROR,         {AppState.SHUTTING_DOWN}),
        (AppState.SHUTTING_DOWN, set()),
    ])
    def test_valid_transitions_property(self, state, expected_targets):
        """valid_transitions() returns the correct frozenset per state."""
        sm = _sm(initial=state)
        assert sm.valid_transitions() == frozenset(expected_targets), (
            f"Wrong valid transitions from {state.name}"
        )


# ---------------------------------------------------------------------------
# Tests: thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Tests verifying that StateMachine is thread-safe."""

    def test_thread_safety(self):
        """20 threads simultaneously calling try_transition must not raise."""
        sm = _sm()
        errors = []

        def worker():
            try:
                sm.try_transition(AppState.CALIBRATING)
                sm.try_transition(AppState.ACTIVE)
                sm.try_transition(AppState.PAUSED)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions occurred in threads: {errors}"
        # Final state must be one of the valid AppState values
        assert sm.state in set(AppState)
