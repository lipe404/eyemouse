
import pytest
from unittest.mock import MagicMock, patch, call
import tkinter as tk
import numpy as np
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from ui.calibration_ui import CalibrationUI
from config import SCREEN_MARGIN, CALIBRATION_POINTS, CALIBRATION_FRAMES_PER_POINT

class TestCalibrationUI:
    @pytest.fixture
    def mock_root(self):
        root = MagicMock()
        # Mocking basic Tk methods
        root.winfo_screenwidth.return_value = 1920
        root.winfo_screenheight.return_value = 1080
        return root

    @pytest.fixture
    def mock_calibration_manager(self):
        return MagicMock()

    @pytest.fixture
    def mock_callbacks(self):
        return {
            'get_latest_gaze': MagicMock(return_value=(0.5, 0.5)),
            'on_complete': MagicMock(),
            'get_latest_frame': MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
        }

    @pytest.fixture
    def calibration_ui(self, mock_root, mock_calibration_manager, mock_callbacks):
        # Mock the entire tk module in ui.calibration_ui
        with patch('ui.calibration_ui.tk') as mock_tk, \
             patch('ui.calibration_ui.cv2') as mock_cv2:
            
            # Setup specific mocks if needed, but MagicMock handles most
            mock_tk.NW = 'nw'
            
            # Setup cv2 mocks
            mock_cv2.resize.return_value = MagicMock()
            mock_cv2.cvtColor.return_value = MagicMock()
            mock_buffer = MagicMock()
            mock_buffer.tobytes.return_value = b'fake_image_data'
            mock_cv2.imencode.return_value = (True, mock_buffer)
            
            ui = CalibrationUI(
                mock_root,
                mock_calibration_manager,
                mock_callbacks['get_latest_gaze'],
                mock_callbacks['on_complete'],
                mock_callbacks['get_latest_frame']
            )
            
            # Attach mocks to the instance for assertion
            ui.mock_tk = mock_tk
            ui.mock_cv2 = mock_cv2
            ui.mock_window = mock_tk.Toplevel.return_value
            # Ensure winfo_exists returns True so loop continues
            ui.mock_window.winfo_exists.return_value = True
            
            ui.mock_canvas = mock_tk.Canvas.return_value
            ui.mock_tk_photoimage = mock_tk.PhotoImage
            yield ui

    def test_init(self, calibration_ui, mock_root):
        """Test initialization of CalibrationUI."""
        assert calibration_ui.root == mock_root
        assert calibration_ui.width == 1920
        assert calibration_ui.height == 1080
        
        # Verify window setup
        calibration_ui.mock_window.title.assert_called_with("Calibração EyeMouse")
        calibration_ui.mock_window.attributes.assert_called_with("-fullscreen", True)
        
        # Verify points generation
        assert len(calibration_ui.points) == CALIBRATION_POINTS
        # Check first point (top-left margin)
        expected_x = SCREEN_MARGIN
        expected_y = SCREEN_MARGIN
        assert calibration_ui.points[0] == (expected_x, expected_y)

    def test_grid_size_fallback(self, mock_root, mock_calibration_manager, mock_callbacks):
        """Test fallback when CALIBRATION_POINTS is not a perfect square."""
        # Patch CALIBRATION_POINTS to a non-square number (e.g., 10)
        with patch('ui.calibration_ui.CALIBRATION_POINTS', 10), \
             patch('ui.calibration_ui.tk') as mock_tk, \
             patch('ui.calibration_ui.cv2'):
            
            ui = CalibrationUI(
                mock_root,
                mock_calibration_manager,
                mock_callbacks['get_latest_gaze'],
                mock_callbacks['on_complete']
            )
            
            # Fallback grid_size is 4, so 4x4 = 16 points
            assert len(ui.points) == 16

    def test_start_sequence(self, calibration_ui):
        """Test start_sequence method."""
        with patch.object(calibration_ui, 'show_point') as mock_show:
            calibration_ui.start_sequence()
            calibration_ui.calib_manager.clear_points.assert_called_once()
            mock_show.assert_called_once()

    def test_show_point_next(self, calibration_ui):
        """Test show_point when there are more points."""
        calibration_ui.current_point_idx = 0
        with patch.object(calibration_ui, 'animate_point') as mock_animate:
            calibration_ui.show_point()
            
            calibration_ui.mock_canvas.delete.assert_has_calls([
                call("target"), call("countdown")
            ])
            mock_animate.assert_called_once()
            args = mock_animate.call_args[0]
            assert args == calibration_ui.points[0]

    def test_show_point_finish(self, calibration_ui):
        """Test show_point when all points are done."""
        calibration_ui.current_point_idx = len(calibration_ui.points)
        with patch.object(calibration_ui, 'finish_calibration') as mock_finish:
            calibration_ui.show_point()
            mock_finish.assert_called_once()

    def test_animate_point_running(self, calibration_ui):
        """Test animate_point while animation is running."""
        # Mock time to simulate mid-animation
        with patch('time.time') as mock_time:
            calibration_ui.animation_start_time = 1000
            mock_time.return_value = 1000 + 0.5 # 0.5s elapsed
            
            calibration_ui.animate_point(100, 100)
            
            calibration_ui.mock_canvas.delete.assert_called()
            calibration_ui.mock_canvas.create_oval.assert_called()
            
            # Check if after was called with 33ms
            args = calibration_ui.mock_window.after.call_args[0]
            assert args[0] == 33
            assert callable(args[1])

    def test_animate_point_finish(self, calibration_ui):
        """Test animate_point when animation is finished."""
        with patch('time.time') as mock_time, \
             patch.object(calibration_ui, 'start_collection') as mock_start:
            
            calibration_ui.animation_start_time = 1000
            mock_time.return_value = 1000 + 2.0 # > 1.5s
            
            calibration_ui.animate_point(100, 100)
            
            calibration_ui.mock_canvas.delete.assert_called()
            mock_start.assert_called_once()
            assert calibration_ui.frames_collected == 0

    def test_start_collection(self, calibration_ui):
        """Test start_collection."""
        with patch.object(calibration_ui, 'collect_loop') as mock_loop:
            calibration_ui.start_collection()
            
            assert calibration_ui.is_collecting is True
            # current_point_idx is passed
            mock_loop.assert_called_once_with(calibration_ui.current_point_idx)

    def test_collect_loop_running(self, calibration_ui):
        """Test collect_loop method collecting data."""
        # Setup
        calibration_ui.current_point_idx = 0
        calibration_ui.frames_collected = 0
        calibration_ui.points = [(100, 100)]
        calibration_ui.is_collecting = True
        
        # Mock gaze returning valid data
        calibration_ui.get_latest_gaze.return_value = (0.5, 0.5)

        calibration_ui.collect_loop(0)
        
        # Check if frames were collected
        assert calibration_ui.frames_collected == 1
        calibration_ui.calib_manager.add_point.assert_called_once()
        
        # Check if scheduled next call
        calibration_ui.mock_window.after.assert_called()
        
    def test_collect_loop_none_gaze(self, calibration_ui):
        """Test collect_loop when gaze is None."""
        calibration_ui.is_collecting = True
        calibration_ui.frames_collected = 0
        calibration_ui.get_latest_gaze.return_value = None
        
        calibration_ui.collect_loop(0)
        
        assert calibration_ui.frames_collected == 0
        calibration_ui.calib_manager.add_point.assert_not_called()
        calibration_ui.mock_window.after.assert_called()

    def test_collect_loop_finished(self, calibration_ui):
        """Test collect_loop when frames for point are collected."""
        calibration_ui.current_point_idx = 0
        calibration_ui.frames_collected = CALIBRATION_FRAMES_PER_POINT
        calibration_ui.points = [(100, 100), (200, 200)]
        calibration_ui.is_collecting = True

        with patch.object(calibration_ui, 'show_point') as mock_show:
            calibration_ui.collect_loop(0)
            
            assert calibration_ui.is_collecting is False
            assert calibration_ui.current_point_idx == 1
            mock_show.assert_called_once()

    def test_update_video_feed(self, calibration_ui):
        """Test update_video_feed."""
        # Reset mocks because __init__ calls update_video_feed once
        calibration_ui.get_latest_frame.reset_mock()
        calibration_ui.mock_tk_photoimage.reset_mock()
        calibration_ui.mock_canvas.create_image.reset_mock()
        calibration_ui.mock_window.after.reset_mock()
        calibration_ui.video_image_id = None # Reset ID to force create_image

        calibration_ui.update_video_feed()
        
        # Should get frame
        calibration_ui.get_latest_frame.assert_called()
        
        # Should create image (using mocks)
        calibration_ui.mock_tk_photoimage.assert_called_once()
        
        # Should update canvas
        calibration_ui.mock_canvas.create_image.assert_called()
        
        # Should schedule next update
        args = calibration_ui.mock_window.after.call_args[0]
        assert args[0] == 33

    def test_update_video_feed_existing_image(self, calibration_ui):
        """Test update_video_feed with existing image ID."""
        calibration_ui.video_image_id = "some_id"
        calibration_ui.mock_canvas.itemconfig.reset_mock()
        calibration_ui.mock_canvas.create_image.reset_mock()
        
        calibration_ui.update_video_feed()
        
        calibration_ui.mock_canvas.itemconfig.assert_called_once()
        calibration_ui.mock_canvas.create_image.assert_not_called()

    def test_update_video_feed_no_window(self, calibration_ui):
        """Test update_video_feed when window is destroyed."""
        calibration_ui.mock_window.winfo_exists.return_value = False
        calibration_ui.get_latest_frame.reset_mock()
        
        calibration_ui.update_video_feed()
        
        calibration_ui.get_latest_frame.assert_not_called()

    def test_finish_calibration(self, calibration_ui):
        """Test finish_calibration."""
        calibration_ui.finish_calibration()
        
        # Verify UI updates
        calibration_ui.mock_canvas.delete.assert_called_with("all")
        calibration_ui.mock_canvas.create_text.assert_called()
        calibration_ui.mock_window.update.assert_called()
        
        # Verify scheduling close
        calibration_ui.mock_window.after.assert_called_with(100, calibration_ui.close)

    def test_close(self, calibration_ui):
        """Test close method."""
        calibration_ui.close()
        
        calibration_ui.mock_window.destroy.assert_called_once()
        calibration_ui.on_complete.assert_called_once_with(cancelled=False)

    def test_on_user_close(self, calibration_ui):
        """Test on_user_close."""
        calibration_ui.is_collecting = True # Should stop
        calibration_ui.on_user_close()
        
        # It does NOT set is_collecting=False explicitly, but assumes destroy handles it
        # calibration_ui.mock_window.destroy.assert_called_once()
        calibration_ui.on_complete.assert_called_once_with(cancelled=True)
