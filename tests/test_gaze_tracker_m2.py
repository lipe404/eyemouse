"""
test_gaze_tracker_m2.py — Testes das novas capacidades do GazeTracker (Milestone 2).
"""
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from gaze_tracker import GazeTracker
from frame_data import FramePacket


class TestGazeTrackerMilestone2:
    @pytest.fixture
    def mock_mediapipe(self):
        with patch('gaze_tracker.python') as mock_python, \
             patch('gaze_tracker.vision') as mock_vision, \
             patch('gaze_tracker.mp') as mock_mp, \
             patch('os.path.exists', return_value=True):

            mock_detector = MagicMock()
            mock_vision.FaceLandmarker.create_from_options.return_value = mock_detector
            mock_vision.RunningMode = MagicMock()
            mock_vision.RunningMode.VIDEO = "VIDEO"
            mock_vision.RunningMode.IMAGE = "IMAGE"
            mock_vision.RunningMode.LIVE_STREAM = "LIVE_STREAM"

            yield mock_mp, mock_detector, mock_vision

    def test_monotonic_timestamps_strictly_increasing(self, mock_mediapipe):
        _, mock_detector, _ = mock_mediapipe
        tracker = GazeTracker("dummy.task", running_mode="VIDEO")

        # Gerar 10 timestamps com o mesmo valor base
        fixed_time = 100.000
        ts1 = tracker._get_monotonic_timestamp_ms(fixed_time)
        ts2 = tracker._get_monotonic_timestamp_ms(fixed_time)
        ts3 = tracker._get_monotonic_timestamp_ms(fixed_time)

        # Mesmo com o mesmo float de entrada, devem ser estritamente crescentes
        assert ts2 > ts1
        assert ts3 > ts2
        assert ts2 == ts1 + 1
        assert ts3 == ts2 + 1
        tracker.close()

    def test_detect_for_video_called_in_video_mode(self, mock_mediapipe):
        _, mock_detector, _ = mock_mediapipe
        tracker = GazeTracker("dummy.task", running_mode="VIDEO")

        # Configurar resultado de detecção no detect_for_video
        mock_result = MagicMock()
        mock_landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(500)]
        mock_result.face_landmarks = [mock_landmarks]
        mock_detector.detect_for_video.return_value = mock_result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        left, right, landmarks = tracker.process_frame(frame, timestamp_ms=1000)

        assert left is not None
        assert right is not None
        assert landmarks is not None
        mock_detector.detect_for_video.assert_called_once()
        tracker.close()

    def test_process_frame_packet(self, mock_mediapipe):
        _, mock_detector, _ = mock_mediapipe
        tracker = GazeTracker("dummy.task", running_mode="VIDEO")

        mock_result = MagicMock()
        mock_landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(500)]
        mock_result.face_landmarks = [mock_landmarks]
        mock_detector.detect_for_video.return_value = mock_result

        now = time.perf_counter()
        packet = FramePacket(
            frame_id=7,
            capture_timestamp=now,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        res = tracker.process_frame_packet(packet)
        assert res.frame_id == 7
        assert res.capture_timestamp == now
        assert res.tracking_valid is True
        assert "avg_iris" in res.features
        tracker.close()

    def test_explicit_close_and_context_manager(self, mock_mediapipe):
        _, mock_detector, _ = mock_mediapipe
        with GazeTracker("dummy.task") as tracker:
            assert tracker._closed is False

        assert tracker._closed is True
        mock_detector.close.assert_called_once()

    def test_draw_debug_respects_debug_draw_flag(self, mock_mediapipe):
        tracker = GazeTracker("dummy.task")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(500)]

        # Com DEBUG_DRAW = False (padrão M2), não desenha e não altera pixels
        with patch('gaze_tracker.DEBUG_DRAW', False):
            tracker.draw_debug(frame, landmarks)
            assert np.all(frame == 0)

        tracker.close()
