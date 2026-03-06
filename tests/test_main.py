
import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import threading
import queue
import numpy as np

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

@pytest.fixture
def mock_app_dependencies():
    # Create mocks for dependencies
    mock_keyboard = MagicMock()
    mock_cv2 = MagicMock()
    mock_cv2.__version__ = '4.0.0'
    mock_cv2.VideoCapture.return_value.isOpened.return_value = True
    
    # Fix camera_loop unpacking error
    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)
    mock_cv2.VideoCapture.return_value.read.return_value = (True, mock_frame)
    mock_cv2.resize.return_value = mock_frame

    mock_mediapipe = MagicMock()
    
    mock_gaze_tracker_module = MagicMock()
    # Fix processing_loop unpacking error
    mock_gaze_tracker_module.GazeTracker.return_value.process_frame.return_value = (MagicMock(), MagicMock(), MagicMock())

    mock_blink_detector_module = MagicMock()
    # Fix blink_detector unpacking error (6 values)
    mock_blink_detector_module.BlinkDetector.return_value.process.return_value = (False, False, False, False, False, (0.3, 0.3))

    mock_calibration_module = MagicMock()
    mock_calibration_module.CalibrationManager.return_value.load_calibration.return_value = True

    mock_mouse_controller_module = MagicMock()
    mock_calibration_ui_module = MagicMock()
    mock_control_panel_module = MagicMock()
    
    # Mock tkinter at module level to avoid GUI
    mock_tk = MagicMock()
    mock_tk.Tk.return_value.withdraw = MagicMock()
    
    # Mock config
    mock_config = MagicMock()
    mock_config.CAMERA_INDEX = 0
    mock_config.CAMERA_WIDTH = 640
    mock_config.CAMERA_HEIGHT = 480
    mock_config.LOG_FILE = "test.log"

    # Patch sys.modules to inject mocks
    with patch.dict(sys.modules, {
        'keyboard': mock_keyboard,
        'cv2': mock_cv2,
        'mediapipe': mock_mediapipe,
        'mediapipe.python.solutions': MagicMock(),
        'mediapipe.tasks': MagicMock(),
        'mediapipe.tasks.python': MagicMock(),
        'mediapipe.tasks.python.vision': MagicMock(),
        'gaze_tracker': mock_gaze_tracker_module,
        'blink_detector': mock_blink_detector_module,
        'calibration': mock_calibration_module,
        'mouse_controller': mock_mouse_controller_module,
        'ui.calibration_ui': mock_calibration_ui_module,
        'ui.control_panel': mock_control_panel_module,
        'tkinter': mock_tk,
        'tkinter.simpledialog': MagicMock(),
        'tkinter.messagebox': MagicMock(),
        # We need to mock config too if it's imported as `from config import *`
        'config': mock_config,
    }):
        # Reload main if it was already imported
        if 'main' in sys.modules:
            del sys.modules['main']
        
        import main
        
        # Patch methods inside main that might use other imports
        with patch('main.simpledialog.askstring', return_value="test_user"), \
             patch('main.messagebox.askyesno', return_value=True), \
             patch('main.sys.exit') as mock_exit:
            
            yield {
                'main': main,
                'cv2': mock_cv2,
                'gaze': mock_gaze_tracker_module.GazeTracker,
                'blink': mock_blink_detector_module.BlinkDetector,
                'calib': mock_calibration_module.CalibrationManager,
                'mouse': mock_mouse_controller_module.MouseController,
                'ui': mock_calibration_ui_module.CalibrationUI,
                'control_panel': mock_control_panel_module.ControlPanel,
                'exit': mock_exit
            }

class TestEyeMouseApp:
    def test_initialization(self, mock_app_dependencies):
        """Test initialization of EyeMouseApp."""
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        # Verify basic setup
        assert app.user_profile == "test_user"
        assert app.running is True
        assert app.is_paused is False
        
        # Verify modules initialized
        mock_app_dependencies['gaze'].assert_called_once()
        mock_app_dependencies['blink'].assert_called_once()
        mock_app_dependencies['calib'].assert_called_with(profile_name="test_user")
        mock_app_dependencies['mouse'].assert_called_once()
        
        # Verify camera setup
        mock_app_dependencies['cv2'].VideoCapture.assert_called()
        
        # Verify calibration loaded
        mock_app_dependencies['calib'].return_value.load_calibration.assert_called_once()
        
        # Cleanup threads to avoid hanging
        app.running = False
        if hasattr(app, 'camera_thread'):
            app.camera_thread.join(timeout=1.0)
        if hasattr(app, 'processing_thread'):
            app.processing_thread.join(timeout=1.0)

    def test_update_smoothing(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        # Test update_smoothing
        app.update_smoothing(0.5)
        # Verify it calls mouse_controller.set_smoothing_alpha
        mock_app_dependencies['mouse'].return_value.set_smoothing_alpha.assert_called_with(0.5)

    def test_show_control_panel(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        # Already shown in __init__
        assert app.control_panel is not None
        mock_app_dependencies['control_panel'].assert_called_once()
        
        # Force close and show again
        app.control_panel = None
        app.show_control_panel()
        
        # Verify instantiated again
        assert app.control_panel is not None
        assert mock_app_dependencies['control_panel'].call_count == 2
        
        # Call again (should not instantiate again)
        app.show_control_panel()
        assert mock_app_dependencies['control_panel'].call_count == 2

    def test_camera_loop_failure(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        app.running = True
        
        # Mock camera to always fail
        mock_app_dependencies['cv2'].VideoCapture.return_value.read.return_value = (False, None)
        
        # Mock time.sleep to run fast
        with patch('main.time.sleep') as mock_sleep, \
             patch('main.messagebox.showerror') as mock_msg:
            
            # This loop runs until failure count reaches 30
            app.camera_loop()
            
            # Verify loop stopped
            assert app.running is False
            
            # Verify root.after called to show error
            # app.root.after(0, lambda: messagebox.showerror(...))
            args, _ = app.root.after.call_args
            assert args[0] == 0
            # Execute the lambda to verify it calls messagebox
            callback = args[1]
            callback()
            mock_msg.assert_called()

    def test_processing_loop_logic(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        # Setup mocks
        app.is_calibrating = False
        app.is_paused = False
        app.running = True
        
        # Important: Set blink_detector.is_calibrating to False explicitly
        # because MagicMock objects are truthy
        app.blink_detector.is_calibrating = False
        
        # Mock frame queue to return one frame then block/stop
        mock_frame = MagicMock()
        mock_frame.shape = (480, 640, 3)
        mock_frame.copy.return_value = mock_frame # Ensure copy returns mock too
        
        # Mock Gaze Tracker
        # left_iris, right_iris, landmarks
        # Need to return numpy arrays for average calculation
        mock_l_iris = np.array([10.0, 10.0])
        mock_r_iris = np.array([20.0, 20.0])
        mock_landmarks = MagicMock()
        
        mock_app_dependencies['gaze'].return_value.process_frame.return_value = (
            mock_l_iris, mock_r_iris, mock_landmarks
        )
        
        # Mock Calibration Manager -> Screen Pos
        mock_app_dependencies['calib'].return_value.map_to_screen.return_value = (100, 200)
        
        # Mock Blink Detector -> Left Click
        # (left, right, double, hold_start, hold_end, ears)
        mock_app_dependencies['blink'].return_value.process.return_value = (
            True, False, False, False, False, (0.3, 0.3)
        )
        
        # Mock Mouse Controller Click to stop loop
        # This simulates stopping after one successful action
        def stop_loop(*args, **kwargs):
            app.running = False
            
        mock_app_dependencies['mouse'].return_value.click.side_effect = stop_loop
        
        # Pre-fill queue with one frame
        app.frame_queue.put(mock_frame)
        
        # Mock cv2.resize since it's used
        mock_app_dependencies['cv2'].resize.return_value = mock_frame
        
        # Disable control panel to simplify
        app.control_panel = None
        
        # Run loop
        # Provide more side_effect values for time.time just in case
        with patch('main.time.time', side_effect=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]): 
             app.processing_loop()
        
        # Assertions
        # 1. Gaze processed
        mock_app_dependencies['gaze'].return_value.process_frame.assert_called()
        
        # 2. Mouse moved
        mock_app_dependencies['mouse'].return_value.move.assert_called_with(100, 200)
        
        # 3. Mouse clicked (left)
        mock_app_dependencies['mouse'].return_value.click.assert_called_with("left")

    def test_blink_calibration_callback(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        app.start_blink_calibration()
        mock_app_dependencies['blink'].return_value.start_calibration.assert_called_once()

    def test_toggle_pause(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        # Initial state
        assert app.is_paused is False
        
        # Toggle
        app.toggle_pause_hotkey()
        assert app.is_paused is True
        
        # Toggle back
        app.toggle_pause_hotkey()
        assert app.is_paused is False
        
        # Cleanup
        app.running = False
        if hasattr(app, 'camera_thread'):
            app.camera_thread.join(timeout=1.0)
        if hasattr(app, 'processing_thread'):
            app.processing_thread.join(timeout=1.0)

    def test_quit_app(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        app.running = True
        
        # Mock threads to avoid joining real threads (though they are daemon)
        app.camera_thread = MagicMock()
        app.processing_thread = MagicMock()
        
        app.quit_app()
        
        assert app.running is False
        mock_app_dependencies['exit'].assert_called_with(0)
        mock_app_dependencies['cv2'].VideoCapture.return_value.release.assert_called()

    def test_start_calibration(self, mock_app_dependencies):
        main_module = mock_app_dependencies['main']
        app = main_module.EyeMouseApp()
        
        app.start_calibration()
        
        assert app.is_calibrating is True
        assert app.is_paused is True
        
        # Verify CalibrationUI was instantiated
        mock_app_dependencies['ui'].assert_called_once()
        
        # Cleanup
        app.running = False
        if hasattr(app, 'camera_thread'):
            app.camera_thread.join(timeout=1.0)
        if hasattr(app, 'processing_thread'):
            app.processing_thread.join(timeout=1.0)
