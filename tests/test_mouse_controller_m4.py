"""
tests/test_mouse_controller_m4.py — Testes para as novas capacidades do MouseController (Milestone 4).
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from mouse_controller import MouseController
from utils.smoothing import OneEuroFilter, KalmanSmoothingFilter, PassThroughFilter


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.screen_size = (1920, 1080)
    backend.button_state = {"left_down": False, "right_down": False}
    backend.is_emergency_stopped = False
    return backend


class TestMouseControllerM4:
    def test_coordinate_stages_separation(self, mock_backend):
        """Verifica se raw_pos, filtered_pos e sent_pos são estritamente separados."""
        ctrl = MouseController(backend=mock_backend, filter_type="ONE_EURO")

        ctrl.move(500.4, 300.6, timestamp=1.0)

        assert ctrl.last_raw_pos == (500.4, 300.6)
        assert isinstance(ctrl.last_filtered_pos[0], float)
        assert isinstance(ctrl.last_filtered_pos[1], float)
        assert ctrl.last_sent_pos == (500, 301)
        mock_backend.move.assert_called_once_with(500, 301)

    def test_precision_mode_reduces_gain(self, mock_backend):
        """No modo de precisão, o ganho em torno da âncora deve ser atenuado pelo precision_factor."""
        ctrl = MouseController(backend=mock_backend, filter_type="NONE")  # NONE para verificar matemática exata

        # Posição inicial
        ctrl.move(500.0, 500.0)
        assert ctrl.last_sent_pos == (500, 500)

        # Ativa modo de precisão com fator 0.3 (30% do movimento)
        ctrl.set_precision_mode(True, factor=0.30)
        assert ctrl.is_precision_mode is True
        assert ctrl.precision_factor == 0.30

        # Movimento de +100px em X e +50px em Y
        # Esperado: 500 + 100 * 0.3 = 530, 500 + 50 * 0.3 = 515
        ctrl.move(600.0, 550.0)
        assert ctrl.last_sent_pos == (530, 515)

    def test_precision_mode_toggle(self, mock_backend):
        ctrl = MouseController(backend=mock_backend)
        assert ctrl.is_precision_mode is False
        new_state = ctrl.toggle_precision_mode()
        assert new_state is True
        assert ctrl.is_precision_mode is True
        new_state2 = ctrl.toggle_precision_mode()
        assert new_state2 is False
        assert ctrl.is_precision_mode is False

    def test_precision_mode_deactivation_clears_anchor(self, mock_backend):
        ctrl = MouseController(backend=mock_backend, filter_type="NONE")
        ctrl.set_precision_mode(True, factor=0.5)
        ctrl.move(100.0, 100.0)
        assert ctrl._precision_anchor is not None

        ctrl.set_precision_mode(False)
        assert ctrl._precision_anchor is None

        # Agora movimento deve ser 100% normal (sem atenuação)
        ctrl.move(200.0, 200.0)
        assert ctrl.last_sent_pos == (200, 200)

    def test_screen_edges_reachable_with_margin_zero(self, mock_backend):
        """Com SCREEN_MARGIN = 0, os cantos exatos da tela (0, 0) e (screen_w-1, screen_h-1) devem ser alcançáveis."""
        ctrl = MouseController(backend=mock_backend, filter_type="NONE")

        # Canto superior esquerdo
        ctrl.move(-50.0, -50.0)
        assert ctrl.last_sent_pos == (0, 0)
        mock_backend.move.assert_called_with(0, 0)

        # Canto inferior direito
        ctrl.move(2500.0, 1500.0)
        assert ctrl.last_sent_pos == (1919, 1079)
        mock_backend.move.assert_called_with(1919, 1079)

    def test_runtime_filter_switching(self, mock_backend):
        """Verifica a alternância entre filtros em tempo de execução."""
        ctrl = MouseController(backend=mock_backend, filter_type="ONE_EURO")
        assert isinstance(ctrl._smoothing, OneEuroFilter)
        assert ctrl.filter_type == "ONE_EURO"

        ctrl.set_filter_type("KALMAN")
        assert isinstance(ctrl._smoothing, KalmanSmoothingFilter)
        assert ctrl.filter_type == "KALMAN"

        ctrl.set_filter_type("NONE")
        assert isinstance(ctrl._smoothing, PassThroughFilter)
        assert ctrl.filter_type == "NONE"

    def test_reset_smoothing_resets_anchor(self, mock_backend):
        ctrl = MouseController(backend=mock_backend)
        ctrl.set_precision_mode(True)
        ctrl.move(100.0, 100.0)
        assert ctrl._precision_anchor is not None

        ctrl.reset_smoothing()
        assert ctrl._precision_anchor is None
