
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from ui.control_panel import ControlPanel

class TestControlPanel:
    @pytest.fixture
    def mock_root(self):
        root = MagicMock()
        root.winfo_screenwidth.return_value = 1920
        return root

    @pytest.fixture
    def mock_callbacks(self):
        return {
            'on_pause_toggle': MagicMock(),
            'on_recalibrate': MagicMock(),
            'on_quit': MagicMock(),
            'update_smoothing': MagicMock(),
            'on_blink_calibrate': MagicMock()
        }

    @pytest.fixture
    def control_panel(self, mock_root, mock_callbacks):
        with patch('ui.control_panel.tk') as mock_tk, \
             patch('ui.control_panel.ttk') as mock_ttk:
            
            # Setup specific mocks for widgets
            mock_tk.Toplevel.return_value.winfo_screenwidth.return_value = 1920
            
            panel = ControlPanel(
                mock_root,
                mock_callbacks['on_pause_toggle'],
                mock_callbacks['on_recalibrate'],
                mock_callbacks['on_quit'],
                mock_callbacks['update_smoothing'],
                mock_callbacks['on_blink_calibrate']
            )
            
            # Attach mocks to instance for assertion
            panel.mock_tk = mock_tk
            panel.mock_ttk = mock_ttk
            panel.mock_window = mock_tk.Toplevel.return_value
            panel.pause_btn = mock_ttk.Button.return_value # This might be overwritten if multiple buttons created
            
            # We need to capture the pause button specifically if possible
            # But since we mock the class, all buttons are instances of the same Mock
            # So checking calls on the class is better, or capturing return values in side_effect
            
            yield panel

    def test_init(self, control_panel, mock_root):
        """Test initialization of ControlPanel."""
        assert control_panel.root == mock_root
        assert control_panel.is_paused is False
        
        # Verify window creation
        control_panel.mock_tk.Toplevel.assert_called_with(mock_root)
        control_panel.mock_window.title.assert_called_with("EyeMouse Control")
        control_panel.mock_window.geometry.assert_called()
        
        # Verify widgets creation
        control_panel.mock_ttk.Label.assert_called()
        control_panel.mock_ttk.Button.assert_called()
        control_panel.mock_ttk.Scale.assert_called()

    def test_toggle_pause(self, control_panel, mock_callbacks):
        """Test pause toggling."""
        # Initial state
        assert control_panel.is_paused is False
        
        # Toggle
        control_panel._toggle_pause()
        
        assert control_panel.is_paused is True
        mock_callbacks['on_pause_toggle'].assert_called_with(True)
        
        # Verify button text update
        # Since we can't easily distinguish which button is which with simple mocks,
        # we check if config was called on A button instance
        control_panel.pause_btn.config.assert_called_with(text="Retomar (Ctrl+Shift+P)")
        
        # Toggle back
        control_panel._toggle_pause()
        
        assert control_panel.is_paused is False
        mock_callbacks['on_pause_toggle'].assert_called_with(False)
        control_panel.pause_btn.config.assert_called_with(text="Pausar (Ctrl+Shift+P)")

    def test_on_scale_change(self, control_panel, mock_callbacks):
        """Test smoothing scale change."""
        control_panel._on_scale_change(0.5)
        mock_callbacks['update_smoothing'].assert_called_with(0.5)
        
        control_panel._on_scale_change("0.8") # String input from tk scale
        mock_callbacks['update_smoothing'].assert_called_with(0.8)

    def test_update_status(self, control_panel):
        """Test status update."""
        # Mock labels specifically if needed, but they are attributes of panel
        # Wait, panel.fps_label is set to mock_ttk.Label() return value
        
        # Since all Label() calls return the same mock object (unless side_effect used),
        # fps_label, eye_status_label etc are the SAME mock object.
        # This is fine for checking if config is called, but we can't distinguish which label got which text easily.
        # However, for 100% coverage we just need to run the code.
        
        control_panel.update_status(30, 0.3, 0.3, 0.2)
        
        # Verify config called on labels
        control_panel.fps_label.config.assert_called()
        control_panel.eye_status_label.config.assert_called()
        control_panel.threshold_label.config.assert_called()
        
        # Check specific text logic
        # Eye closed (EAR < threshold)
        control_panel.update_status(30, 0.1, 0.1, 0.2)
        
        # Verify call args contains asterisk
        # Since all labels are the same mock, we check call_args_list
        calls = control_panel.eye_status_label.config.call_args_list
        found_asterisk = False
        for call in calls:
            kwargs = call[1]
            if 'text' in kwargs and "*" in kwargs['text']:
                found_asterisk = True
                break
        
        assert found_asterisk

    def test_callbacks_wiring(self, control_panel, mock_callbacks):
        """Test if buttons are wired to correct callbacks."""
        # This is hard to test with simple mocks without inspecting command arg of Button calls.
        # We can inspect call_args_list of Button.
        
        # Iterate over all Button calls
        calls = control_panel.mock_ttk.Button.call_args_list
        
        # Check if we find our callbacks in the commands
        found_recalibrate = False
        found_quit = False
        found_blink_calibrate = False
        
        for call in calls:
            kwargs = call[1]
            cmd = kwargs.get('command')
            if cmd == control_panel.on_recalibrate:
                found_recalibrate = True
            elif cmd == control_panel.on_quit:
                found_quit = True
            elif cmd == control_panel.on_blink_calibrate:
                found_blink_calibrate = True
                
        assert found_recalibrate
        assert found_quit
        assert found_blink_calibrate
