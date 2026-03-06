import pytest
import os
import sys
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from gaze_tracker import GazeTracker

class TestGazeTracker:
    @pytest.fixture
    def mock_mediapipe(self):
        with patch('gaze_tracker.python') as mock_python, \
             patch('gaze_tracker.vision') as mock_vision, \
             patch('gaze_tracker.mp') as mock_mp:
            
            # Setup mock detector
            mock_detector = MagicMock()
            mock_vision.FaceLandmarker.create_from_options.return_value = mock_detector
            
            yield mock_mp, mock_detector

    @pytest.fixture
    def tracker(self, mock_mediapipe):
        with patch('os.path.exists', return_value=True):
            return GazeTracker("dummy.task")

    def test_initialization(self, tracker, mock_mediapipe):
        mock_mp, mock_detector = mock_mediapipe
        assert tracker.detector == mock_detector
        assert len(tracker.LEFT_IRIS) > 0
        assert len(tracker.RIGHT_IRIS) > 0

    def test_initialization_file_not_found(self):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                GazeTracker("missing.task")

    def test_get_iris_position(self, tracker):
        # Criar landmarks fictícios
        landmarks = [MagicMock() for _ in range(500)]
        indices = [0, 1, 2]
        
        # Configurar coordenadas
        landmarks[0].x, landmarks[0].y = 0.1, 0.1
        landmarks[1].x, landmarks[1].y = 0.2, 0.2
        landmarks[2].x, landmarks[2].y = 0.3, 0.3
        
        center = tracker.get_iris_position(landmarks, indices, 100, 100)
        
        # Média: (0.1+0.2+0.3)/3 = 0.2
        assert np.isclose(center[0], 0.2)
        assert np.isclose(center[1], 0.2)

    def test_process_frame_success(self, tracker, mock_mediapipe):
        mock_mp, mock_detector = mock_mediapipe
        
        # Mock frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock detection result
        mock_result = MagicMock()
        mock_landmarks = [MagicMock() for _ in range(500)]
        # Configurar landmarks da íris para evitar erro no get_iris_position
        for lm in mock_landmarks:
            lm.x = 0.5
            lm.y = 0.5
            
        mock_result.face_landmarks = [mock_landmarks]
        mock_detector.detect.return_value = mock_result
        
        left, right, landmarks = tracker.process_frame(frame)
        
        assert left is not None
        assert right is not None
        assert landmarks is not None
        mock_detector.detect.assert_called_once()

    def test_process_frame_no_face(self, tracker, mock_mediapipe):
        mock_mp, mock_detector = mock_mediapipe
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        mock_result = MagicMock()
        mock_result.face_landmarks = [] # Nenhuma face
        mock_detector.detect.return_value = mock_result
        
        left, right, landmarks = tracker.process_frame(frame)
        
        assert left is None
        assert right is None
        assert landmarks is None

    def test_draw_debug(self, tracker):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [MagicMock() for _ in range(500)]
        for lm in landmarks:
            lm.x = 0.5
            lm.y = 0.5
            
        # Não deve crashar
        tracker.draw_debug(frame, landmarks)
        
        # Verificar se cv2 desenhou algo (difícil de testar diretamente sem mockar cv2, 
        # mas se não crashou é bom sinal. Podemos mockar cv2 se necessário, mas 
        # teste de integração visual é melhor manual ou snapshot)
        
    def test_draw_debug_no_landmarks(self, tracker):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.draw_debug(frame, None)
        # Não faz nada, não crasha
