import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from mouse_controller import MouseController

class TestMouseController:
    @pytest.fixture
    def mock_pyautogui(self):
        with patch('mouse_controller.pyautogui') as mock:
            mock.size.return_value = (1920, 1080)
            yield mock

    @pytest.fixture
    def mock_winsound(self):
        with patch('mouse_controller.winsound') as mock:
            yield mock

    @pytest.fixture
    def controller(self, mock_pyautogui, mock_winsound):
        return MouseController()

    def test_initialization(self, controller, mock_pyautogui):
        mock_pyautogui.size.assert_called_once()
        assert controller.screen_w == 1920
        assert controller.screen_h == 1080
        assert controller.is_dragging is False
        assert controller.smoothing is not None

    def test_move(self, controller, mock_pyautogui):
        # Primeiro movimento
        controller.move(100, 200)
        
        # O filtro de suavização deve ter sido chamado
        # O pyautogui.moveTo deve ter sido chamado com coordenadas
        # Nota: as coordenadas exatas dependem do SmoothingFilter
        mock_pyautogui.moveTo.assert_called()
        args, _ = mock_pyautogui.moveTo.call_args
        assert isinstance(args[0], (int, float))
        assert isinstance(args[1], (int, float))

    def test_move_bounds(self, controller, mock_pyautogui):
        # Tentar mover para fora da tela (negativo)
        controller.move(-100, -100)
        args, _ = mock_pyautogui.moveTo.call_args
        x, y = args
        # Deve respeitar SCREEN_MARGIN (padrão 50)
        assert x >= 50
        assert y >= 50
        
        # Tentar mover para fora da tela (muito grande)
        controller.move(3000, 3000)
        args, _ = mock_pyautogui.moveTo.call_args
        x, y = args
        assert x <= 1920 - 50
        assert y <= 1080 - 50

    def test_click_left(self, controller, mock_pyautogui):
        with patch.object(controller, '_play_sound') as mock_sound:
            controller.click("left")
            mock_pyautogui.click.assert_called_with(button="left")
            mock_sound.assert_called_with(1000, 50)

    def test_click_right(self, controller, mock_pyautogui):
        with patch.object(controller, '_play_sound') as mock_sound:
            controller.click("right")
            mock_pyautogui.click.assert_called_with(button="right")
            mock_sound.assert_called_with(500, 50)

    def test_double_click(self, controller, mock_pyautogui):
        with patch.object(controller, '_play_sound') as mock_sound:
            controller.double_click()
            mock_pyautogui.doubleClick.assert_called_once()
            mock_sound.assert_called_with(1500, 50)

    def test_drag(self, controller, mock_pyautogui):
        with patch.object(controller, '_play_sound') as mock_sound:
            # Start drag
            controller.start_drag()
            assert controller.is_dragging is True
            mock_pyautogui.mouseDown.assert_called_once()
            mock_sound.assert_called_with(800, 200)
            
            # Tentar start drag de novo (não deve fazer nada)
            mock_pyautogui.mouseDown.reset_mock()
            controller.start_drag()
            mock_pyautogui.mouseDown.assert_not_called()
            
            # Stop drag
            controller.stop_drag()
            assert controller.is_dragging is False
            mock_pyautogui.mouseUp.assert_called_once()
            mock_sound.assert_called_with(600, 100)
            
            # Tentar stop drag de novo (não deve fazer nada)
            mock_pyautogui.mouseUp.reset_mock()
            controller.stop_drag()
            mock_pyautogui.mouseUp.assert_not_called()

    def test_set_smoothing_alpha(self, controller):
        controller.set_smoothing_alpha(0.8)
        assert controller.smoothing.current_alpha == 0.8

    def test_play_sound(self, controller, mock_winsound):
        """Test sound playback logic."""
        # We need to verify that a thread is started
        with patch('threading.Thread') as mock_thread:
            controller._play_sound(440, 100)
            mock_thread.assert_called_once()
            
            # Verify target function calls winsound
            args, kwargs = mock_thread.call_args
            target = kwargs.get('target')
            
            # Execute the target function
            target()
            mock_winsound.Beep.assert_called_with(440, 100)
            
            # Test exception handling inside thread
            mock_winsound.Beep.side_effect = Exception("Sound error")
            try:
                target() # Should not raise exception
            except Exception:
                pytest.fail("Exception should be caught inside _play_sound thread")

    def test_import_error_winsound(self):
        """Test behavior when winsound is not available."""
        # Remove mouse_controller from sys.modules to force reload
        if 'mouse_controller' in sys.modules:
            del sys.modules['mouse_controller']
            
        # Mock sys.modules so 'winsound' import fails
        with patch.dict(sys.modules, {'winsound': None}):
            # We can't just set it to None, import will succeed if None is in sys.modules? 
            # No, if it's None, it might mean "module not found" in some contexts or just None.
            # Better to raise ImportError using side_effect on builtins.__import__?
            # Or just ensure it's not in sys.modules and we can't import it?
            # Windows usually has it.
            
            # Let's try patching builtins.__import__ only for winsound
            import builtins
            original_import = builtins.__import__
            
            def side_effect(name, *args, **kwargs):
                if name == 'winsound':
                    raise ImportError("No winsound")
                return original_import(name, *args, **kwargs)
                
            with patch('builtins.__import__', side_effect=side_effect):
                import mouse_controller
                assert mouse_controller.winsound is None
                
                controller = mouse_controller.MouseController()
                # _play_sound should simply do nothing if winsound is None
                with patch('threading.Thread') as mock_thread:
                    controller._play_sound(440, 100)
                    mock_thread.assert_not_called()

