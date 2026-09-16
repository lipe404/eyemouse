"""
test_camera_capture.py — Testes unitários para o módulo CameraCapture.
"""
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from camera_capture import CameraCapture, fourcc_to_str, str_to_fourcc, BACKEND_MAP


class TestCameraCaptureUtils:
    def test_fourcc_conversion(self):
        mjpg_val = str_to_fourcc("MJPG")
        assert mjpg_val != 0
        decoded = fourcc_to_str(mjpg_val)
        assert decoded == "MJPG"

    def test_str_to_fourcc_invalid(self):
        assert str_to_fourcc("XYZ") == 0


class TestCameraCapture:
    @pytest.fixture
    def mock_cv2(self):
        with patch('camera_capture.cv2.VideoCapture') as mock_vc_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                3: 640.0,   # CAP_PROP_FRAME_WIDTH
                4: 480.0,   # CAP_PROP_FRAME_HEIGHT
                5: 30.0,    # CAP_PROP_FPS
                6: float(str_to_fourcc("MJPG")), # CAP_PROP_FOURCC
            }.get(prop, 0.0)
            mock_cap.set.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_vc_cls.return_value = mock_cap
            yield mock_vc_cls, mock_cap

    def test_initialization_parameters(self, mock_cv2):
        mock_vc_cls, mock_cap = mock_cv2
        cam = CameraCapture(
            camera_index=0,
            width=640,
            height=480,
            fps=30,
            backend="DSHOW",
            fourcc="MJPG",
        )
        assert cam.metadata["actual_width"] == 640
        assert cam.metadata["actual_height"] == 480
        assert cam.metadata["backend"] == "DSHOW"
        assert len(cam.rejected_configurations) == 0
        cam.release()

    def test_rejected_resolution_recorded(self, mock_cv2):
        mock_vc_cls, mock_cap = mock_cv2
        # Câmera ignora pedido de 1280x720 e devolve 640x480
        mock_cap.get.side_effect = lambda prop: {
            3: 640.0,
            4: 480.0,
            5: 30.0,
        }.get(prop, 0.0)

        cam = CameraCapture(
            camera_index=0,
            width=1280,
            height=720,
        )
        assert len(cam.rejected_configurations) > 0
        assert any("ignorada" in rej for rej in cam.rejected_configurations)
        cam.release()

    def test_buffersize_rejection_handled_defensively(self, mock_cv2):
        mock_vc_cls, mock_cap = mock_cv2
        # Simular backend que rejeita CAP_PROP_BUFFERSIZE
        mock_cap.set.side_effect = lambda prop, val: False if prop == 38 else True

        cam = CameraCapture(camera_index=0)
        assert any("BUFFERSIZE" in rej for rej in cam.rejected_configurations)
        cam.release()

    def test_capture_thread_and_get_latest_frame(self, mock_cv2):
        mock_vc_cls, mock_cap = mock_cv2
        cam = CameraCapture(camera_index=0)
        cam.start()

        # Aguardar recebimento de frame
        packet = cam.get_latest_frame(timeout=0.5)
        assert packet is not None
        assert packet.frame_id >= 1
        assert packet.image.shape == (480, 640, 3)

        cam.release()

    def test_context_manager(self, mock_cv2):
        with CameraCapture(camera_index=0) as cam:
            assert cam._running is True
            packet = cam.get_latest_frame(timeout=0.5)
            assert packet is not None
        assert cam._running is False
