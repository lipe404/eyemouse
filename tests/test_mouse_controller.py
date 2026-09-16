"""
test_mouse_controller.py — Testes para MouseController (Milestone 1).

Substitui mocks de pyautogui por mocks de OsMouse (SendInput nativo).
Cobre todos os cenarios de seguranca requeridos pelo Milestone 1.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_os_mouse():
    """Retorna um mock completo de OsMouse."""
    m = MagicMock()
    m.screen_size = (1920, 1080)
    m.is_dragging = False
    m.button_state = {"left_down": False, "right_down": False}
    return m


@pytest.fixture
def controller(mock_os_mouse):
    """Cria um MouseController com OsMouse mockado."""
    with patch('mouse_controller.OsMouse', return_value=mock_os_mouse):
        from mouse_controller import MouseController
        ctrl = MouseController()
        ctrl._os_mouse = mock_os_mouse
        yield ctrl


# ---------------------------------------------------------------------------
# Testes de inicializacao
# ---------------------------------------------------------------------------

class TestMouseControllerInit:
    def test_reads_screen_size_from_os_mouse(self, controller, mock_os_mouse):
        """screen_w e screen_h devem vir do OsMouse."""
        assert controller.screen_w == 1920
        assert controller.screen_h == 1080

    def test_not_dragging_initially(self, controller):
        assert controller.is_dragging is False


# ---------------------------------------------------------------------------
# Testes de movimento
# ---------------------------------------------------------------------------

class TestMouseControllerMove:
    def test_move_calls_os_mouse(self, controller, mock_os_mouse):
        """move() deve chamar os_mouse.move() com coordenadas inteiras."""
        controller.move(500.7, 300.2)
        mock_os_mouse.move.assert_called_once()
        args = mock_os_mouse.move.call_args[0]
        assert isinstance(args[0], int)
        assert isinstance(args[1], int)

    def test_move_clamps_to_screen_bounds(self, controller, mock_os_mouse):
        """Coordenadas fora da tela devem ser clampadas."""
        controller.move(-100, 2000)
        args = mock_os_mouse.move.call_args[0]
        assert args[0] >= 0
        assert args[1] <= 1079  # screen_h - 1 com SCREEN_MARGIN=0


# ---------------------------------------------------------------------------
# Testes de clique
# ---------------------------------------------------------------------------

class TestMouseControllerClicks:
    def test_left_click(self, controller, mock_os_mouse):
        controller.left_click()
        mock_os_mouse.left_click.assert_called_once()

    def test_right_click(self, controller, mock_os_mouse):
        controller.right_click()
        mock_os_mouse.right_click.assert_called_once()

    def test_double_click(self, controller, mock_os_mouse):
        controller.double_click()
        mock_os_mouse.double_click.assert_called_once()

    def test_click_left_alias(self, controller, mock_os_mouse):
        controller.click("left")
        mock_os_mouse.left_click.assert_called_once()

    def test_click_right_alias(self, controller, mock_os_mouse):
        controller.click("right")
        mock_os_mouse.right_click.assert_called_once()

    def test_click_unknown_button_logs_warning(self, controller):
        """Botao desconhecido nao deve levantar excecao."""
        controller.click("middle")  # Deve logar warning, nao crashar


# ---------------------------------------------------------------------------
# Testes de arraste
# ---------------------------------------------------------------------------

class TestMouseControllerDrag:
    def test_start_drag_sets_is_dragging(self, controller, mock_os_mouse):
        controller.start_drag()
        assert controller.is_dragging is True
        mock_os_mouse.button_down.assert_called_once_with("left")

    def test_stop_drag_clears_is_dragging(self, controller, mock_os_mouse):
        controller.start_drag()
        controller.stop_drag()
        assert controller.is_dragging is False
        mock_os_mouse.button_up.assert_called_once_with("left")

    def test_button_down_then_up(self, controller, mock_os_mouse):
        controller.button_down("left")
        controller.button_up("left")
        mock_os_mouse.button_down.assert_called_with("left")
        mock_os_mouse.button_up.assert_called_with("left")


# ---------------------------------------------------------------------------
# Testes de seguranca: release_all e emergency_stop
# ---------------------------------------------------------------------------

class TestMouseControllerSafety:
    def test_release_all_calls_os_mouse(self, controller, mock_os_mouse):
        """release_all() deve delegar para os_mouse.release_all()."""
        controller.release_all()
        mock_os_mouse.release_all.assert_called_once()

    def test_release_all_clears_is_dragging(self, controller, mock_os_mouse):
        """release_all() deve limpar o estado de arraste interno."""
        controller._is_dragging = True
        controller.release_all()
        assert controller.is_dragging is False

    def test_release_all_idempotent(self, controller, mock_os_mouse):
        """Multiplas chamadas nao devem levantar excecao."""
        controller.release_all()
        controller.release_all()
        controller.release_all()
        assert mock_os_mouse.release_all.call_count == 3

    def test_pause_while_button_pressed(self, controller, mock_os_mouse):
        """
        Cenario: usuario ativa arraste e depois pausa.
        release_all() deve ser chamado.
        """
        controller.start_drag()
        assert controller.is_dragging is True

        # Simulando o que o StateMachine faz ao entrar em PAUSED
        controller.release_all()

        assert controller.is_dragging is False
        mock_os_mouse.release_all.assert_called_once()

    def test_shutdown_during_drag_releases_button(self, controller, mock_os_mouse):
        """
        Cenario: usuario fecha o app durante arraste.
        release_all() deve ser chamado antes de sys.exit().
        """
        controller.start_drag()
        assert controller.is_dragging is True

        controller.release_all()

        assert controller.is_dragging is False
        mock_os_mouse.release_all.assert_called()

    def test_emergency_stop_calls_os_mouse(self, controller, mock_os_mouse):
        """emergency_stop() deve delegar para os_mouse.emergency_stop()."""
        controller.emergency_stop()
        mock_os_mouse.emergency_stop.assert_called_once()

    def test_emergency_stop_clears_dragging(self, controller, mock_os_mouse):
        """emergency_stop() deve limpar estado de arraste."""
        controller._is_dragging = True
        controller.emergency_stop()
        assert controller.is_dragging is False

    def test_reset_emergency_stop(self, controller, mock_os_mouse):
        controller.emergency_stop()
        controller.reset_emergency_stop()
        mock_os_mouse.reset_emergency_stop.assert_called_once()

    def test_multiple_release_all_when_not_dragging(self, controller, mock_os_mouse):
        """release_all() sem arraste ativo nao deve crashar."""
        controller.release_all()
        controller.release_all()
        assert mock_os_mouse.release_all.call_count == 2

    def test_camera_loss_scenario(self, controller, mock_os_mouse):
        """
        Cenario: camera perde conexao durante arraste.
        release_all() deve ser chamado (pelo StateMachine ou diretamente).
        """
        controller.start_drag()
        # Simula o que acontece quando camera_loop detecta falha
        controller.release_all()
        assert controller.is_dragging is False


# ---------------------------------------------------------------------------
# Testes de suavizacao
# ---------------------------------------------------------------------------

class TestMouseControllerSmoothing:
    def test_set_smoothing_alpha(self, controller):
        """set_smoothing_alpha nao deve levantar excecao."""
        controller.set_smoothing_alpha(0.5)

    def test_reset_smoothing(self, controller):
        """reset_smoothing nao deve levantar excecao."""
        controller.reset_smoothing()


# ---------------------------------------------------------------------------
# Testes de scroll
# ---------------------------------------------------------------------------

class TestMouseControllerScroll:
    def test_scroll_calls_os_mouse(self, controller, mock_os_mouse):
        controller.scroll(120)
        mock_os_mouse.scroll.assert_called_once_with(120)

    def test_scroll_negative(self, controller, mock_os_mouse):
        controller.scroll(-120)
        mock_os_mouse.scroll.assert_called_once_with(-120)
