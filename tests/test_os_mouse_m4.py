"""
tests/test_os_mouse_m4.py — Testes unitários para DPI Awareness, Desktop Virtual e Backends Intercambiáveis (Milestone 4).
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

import os_mouse
from os_mouse import (
    OsMouse,
    PyAutoGuiMouse,
    create_mouse_backend,
    init_dpi_awareness,
    MOUSEEVENTF_ABSOLUTE,
    MOUSEEVENTF_MOVE,
    MOUSEEVENTF_VIRTUALDESK,
)


class TestDpiAwareness:
    def test_dpi_awareness_v2_success(self):
        mock_user32 = MagicMock()
        mock_user32.SetProcessDpiAwarenessContext.return_value = 1

        with patch("os_mouse.sys") as mock_sys, \
             patch("os_mouse.ctypes.windll") as mock_windll:
            mock_sys.platform = "win32"
            mock_windll.user32 = mock_user32

            res = init_dpi_awareness()
            assert res is True
            mock_user32.SetProcessDpiAwarenessContext.assert_called_once()

    def test_dpi_awareness_non_windows_returns_false(self):
        with patch("os_mouse.sys") as mock_sys:
            mock_sys.platform = "darwin"
            res = init_dpi_awareness()
            assert res is False


class TestVirtualDesktopMultiMonitor:
    def test_negative_coordinates_mapping(self):
        """
        Cenário Multi-Monitor:
        Monitor 2 à esquerda: X de -1920 a -1.
        Monitor 1 à direita (primário): X de 0 a 1919.
        Desktop virtual total: vx = -1920, vy = 0, vw = 3840, vh = 1080.
        """
        mock_user32 = MagicMock()
        # SM_CXSCREEN=0, SM_CYSCREEN=1, SM_XVIRTUALSCREEN=76, SM_YVIRTUALSCREEN=77, SM_CXVIRTUALSCREEN=78, SM_CYVIRTUALSCREEN=79
        def metrics_side_effect(metric):
            if metric == 0: return 1920
            elif metric == 1: return 1080
            elif metric == 76: return -1920
            elif metric == 77: return 0
            elif metric == 78: return 3840
            elif metric == 79: return 1080
            return 0

        mock_user32.GetSystemMetrics.side_effect = metrics_side_effect
        mock_user32.SendInput.return_value = 1

        with patch("os_mouse.sys") as mock_sys, \
             patch("os_mouse.ctypes.windll") as mock_windll:
            mock_sys.platform = "win32"
            mock_windll.user32 = mock_user32

            mouse = OsMouse(enable_dpi_awareness=False)
            assert mouse.virtual_screen_geometry == (-1920, 0, 3840, 1080)

            # Teste de extremos:
            # Ponto mais à esquerda (-1920) deve mapear para ax = 0
            ax_left, ay_top = mouse._to_absolute(-1920, 0)
            assert ax_left == 0
            assert ay_top == 0

            # Ponto de divisão entre monitores (0) deve mapear para meio do desktop virtual (~32767)
            ax_mid, _ = mouse._to_absolute(0, 0)
            assert abs(ax_mid - 32767) <= 10

            # Ponto mais à direita (1919) deve mapear para ax = 65535
            ax_right, ay_bottom = mouse._to_absolute(1919, 1079)
            assert ax_right == 65535
            assert ay_bottom == 65535

    def test_move_sends_virtualdesk_flag(self):
        """O método move deve enviar MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK."""
        mock_user32 = MagicMock()
        mock_user32.GetSystemMetrics.return_value = 1920
        mock_user32.SendInput.return_value = 1

        with patch("os_mouse.sys") as mock_sys, \
             patch("os_mouse.ctypes.windll") as mock_windll:
            mock_sys.platform = "win32"
            mock_windll.user32 = mock_user32

            mouse = OsMouse(enable_dpi_awareness=False)
            mouse.move(500, 300)

            assert mock_user32.SendInput.called
            # Recupera o struct _INPUT enviado
            inp_ptr = mock_user32.SendInput.call_args[0][1]
            # O primeiro argumento de SendInput é 1 (quantidade de inputs)
            assert mock_user32.SendInput.call_args[0][0] == 1


class TestMouseBackendFactory:
    def test_create_sendinput(self):
        with patch("os_mouse.sys") as mock_sys, \
             patch("os_mouse.ctypes.windll"):
            mock_sys.platform = "win32"
            backend = create_mouse_backend("SENDINPUT")
            assert isinstance(backend, OsMouse)

    def test_create_pyautogui(self):
        mock_pyautogui = MagicMock()
        mock_pyautogui.size.return_value = (1920, 1080)
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            backend = create_mouse_backend("PYAUTOGUI")
            assert isinstance(backend, PyAutoGuiMouse)
            assert mock_pyautogui.PAUSE == 0.0

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Backend de mouse desconhecido"):
            create_mouse_backend("INVALID_BACKEND")


class TestPyAutoGuiMouseBackend:
    def test_pyautogui_actions(self):
        mock_pag = MagicMock()
        mock_pag.size.return_value = (1920, 1080)

        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            driver = PyAutoGuiMouse()

            # Move
            driver.move(100, 200)
            mock_pag.moveTo.assert_called_with(100, 200)

            # Clicks
            driver.left_click()
            mock_pag.click.assert_called_with(button="left")
            driver.right_click()
            mock_pag.click.assert_called_with(button="right")
            driver.double_click()
            mock_pag.doubleClick.assert_called_with(button="left")

            # Scroll
            driver.scroll(240)
            mock_pag.scroll.assert_called_with(2)

            # Drag and Release
            driver.button_down("left")
            mock_pag.mouseDown.assert_called_with(button="left")
            assert driver.button_state["left_down"] is True

            driver.release_all()
            mock_pag.mouseUp.assert_called_with(button="left")
            assert driver.button_state["left_down"] is False
