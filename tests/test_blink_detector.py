
import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from blink_detector import BlinkDetector

class TestBlinkDetector:
    @pytest.fixture
    def detector(self):
        return BlinkDetector()

    def create_mock_landmarks(self, detector=None, eyes_open=True, left_open=None, right_open=None):
        """Cria landmarks simulados para olhos abertos ou fechados"""
        if detector is None:
            detector = BlinkDetector()

        if left_open is None: left_open = eyes_open
        if right_open is None: right_open = eyes_open

        landmarks = [MagicMock() for _ in range(468)] # 468 landmarks no Face Mesh
        
        # Índices chave para o cálculo do EAR
        # Esquerdo: [362, 385, 387, 263, 373, 380]
        # Direito: [33, 160, 158, 133, 153, 144]
        
        # Padrão: Olhos abertos (EAR ~ 0.3)
        # Altura (v1, v2) = 6, Largura (h) = 20 -> (6+6)/(2*20) = 0.3
        
        h = 20
        # v = 6 if eyes_open else 2 # 2px altura se fechado -> EAR ~ 0.1
        
        # Configurar coordenadas (simplificado, apenas relativo)
        # Vamos configurar apenas os landmarks que importam
        
        def set_eye_landmarks(indices, offset_x, is_open):
            v = 6 if is_open else 2
            p1, p2, p3, p4, p5, p6 = indices
            # p1 (canto esq), p4 (canto dir) -> Largura
            landmarks[p1].x = (offset_x) / 100.0
            landmarks[p1].y = 50 / 100.0
            landmarks[p4].x = (offset_x + h) / 100.0
            landmarks[p4].y = 50 / 100.0
            
            # p2, p6 (vertical 1)
            landmarks[p2].x = (offset_x + h/3) / 100.0
            landmarks[p2].y = (50 - v/2) / 100.0
            landmarks[p6].x = (offset_x + h/3) / 100.0
            landmarks[p6].y = (50 + v/2) / 100.0
            
            # p3, p5 (vertical 2)
            landmarks[p3].x = (offset_x + 2*h/3) / 100.0
            landmarks[p3].y = (50 - v/2) / 100.0
            landmarks[p5].x = (offset_x + 2*h/3) / 100.0
            landmarks[p5].y = (50 + v/2) / 100.0

        set_eye_landmarks(detector.LEFT_EYE_IDXS, 10, left_open)
        set_eye_landmarks(detector.RIGHT_EYE_IDXS, 60, right_open)
        
        return landmarks

    def test_calculate_ear(self, detector):
        landmarks = self.create_mock_landmarks(detector, eyes_open=True)
        # Usando img_w=100, img_h=100 para simplificar coordenadas = pixels
        ear = detector.calculate_ear(landmarks, detector.LEFT_EYE_IDXS, 100, 100)
        
        # Altura 6, Largura 20 -> EAR 0.3
        assert 0.29 < ear < 0.31
        
        landmarks_closed = self.create_mock_landmarks(detector, eyes_open=False)
        ear_closed = detector.calculate_ear(landmarks_closed, detector.LEFT_EYE_IDXS, 100, 100)
        
        # Altura 2, Largura 20 -> EAR 0.1
        assert 0.09 < ear_closed < 0.11

    def test_calculate_ear_zero_division(self, detector):
        landmarks = self.create_mock_landmarks(detector)
        # Forçar largura zero: p1.x == p4.x
        idx1 = detector.LEFT_EYE_IDXS[0]
        idx4 = detector.LEFT_EYE_IDXS[3]
        landmarks[idx1].x = 0.5
        landmarks[idx4].x = 0.5 # Mesma posição x
        landmarks[idx1].y = 0.5
        landmarks[idx4].y = 0.5 # Mesma posição y
        
        ear = detector.calculate_ear(landmarks, detector.LEFT_EYE_IDXS, 100, 100)
        assert ear == 0.0

    def test_process_grace_period(self, detector):
        landmarks = self.create_mock_landmarks(detector, eyes_open=True)
        # Durante o período de graça, deve retornar tudo False
        # GRACE_PERIOD_FRAMES = 30
        
        for _ in range(10):
            res = detector.process(landmarks, 100, 100)
            assert res[0] is False # left_action
            
        assert detector.face_frames == 10

    def test_calibration(self, detector):
        # Mock time.time to control execution flow
        with patch('blink_detector.time.time') as mock_time:
            start_t = 1000.0
            mock_time.return_value = start_t
            
            detector.start_calibration(duration=0.1)
            assert detector.is_calibrating is True
            
            landmarks = self.create_mock_landmarks(detector, eyes_open=True)
            
            # Processar durante calibração (time + 0.05)
            mock_time.return_value = start_t + 0.05
            res = detector.process(landmarks, 100, 100)
            assert detector.is_calibrating is True
            assert len(detector.calibration_samples) == 1
            
            # Próximo process deve finalizar calibração (time + 0.15)
            mock_time.return_value = start_t + 0.15
            res = detector.process(landmarks, 100, 100)
            assert detector.is_calibrating is False
            # Threshold deve ser ajustado (0.3 * 0.75 = 0.225)
            assert 0.22 < detector.ear_threshold < 0.23

    def test_blink_detection(self, detector):
        # Avançar período de graça
        landmarks_open = self.create_mock_landmarks(detector, eyes_open=True)
        for _ in range(detector.GRACE_PERIOD_FRAMES + 5):
            detector.process(landmarks_open, 100, 100)
            
        # Simular piscada (Olhos Fechados por ~4 frames)
        # Importante: Fechar APENAS o olho esquerdo para evitar deteção de duplo clique (que anula a ação individual)
        landmarks_closed = self.create_mock_landmarks(detector, left_open=False, right_open=True)
        
        # Frame 1-5: Fechando (Smoothing delay)
        # O histórico tem tamanho 5. Precisa de ~3 frames para o EAR médio cair abaixo do threshold (0.22)
        # Open=0.3, Closed=0.1
        
        frames_closed = 0
        for _ in range(5):
             detector.process(landmarks_closed, 100, 100)
             if detector.left_closed_frames > 0:
                 frames_closed += 1
        
        # Verifica se detectou o fechamento
        assert detector.left_closed_frames > 0
        
        # Agora abrir o olho
        # Frame: Abriu (Deve disparar ação)
        
        # IMPORTANTE: Forçar cooldown para garantir que o clique seja aceito
        # Usamos 0.0 para garantir que current_time - last_time > COOLDOWN
        detector.last_left_blink_time = 0 
        
        # Precisa de frames abertos para subir o EAR médio acima do threshold
        # E disparar a ação NA TRANSIÇÃO (quando left_closed vira False)
        
        action_triggered = False
        
        # Garantir que não seja detectado como reflexo
        # Aumentar o limite de reflexo para este teste
        with patch('blink_detector.BLINK_REFLEX_OPEN_SPEED', 1.0):
            # Processar alguns frames abertos
            for i in range(10): # Aumentado para 10 para garantir transição do buffer
                res = detector.process(landmarks_open, 100, 100)
                import blink_detector as bd
                print(f"Frame {i}: Closed Frames={detector.left_closed_frames}, Action={res[0]}")
                print(f"  Debug: Threshold={detector.ear_threshold}, Reflex={detector.left_is_reflex}")
                print(f"  Debug: MinFrames={bd.BLINK_MIN_FRAMES}, MaxFrames={bd.BLINK_MAX_FRAMES}")
                print(f"  Debug: Cooldown={bd.BLINK_COOLDOWN_SEC}, LastTime={detector.last_left_blink_time}")
                if res[0]: # left_action
                    action_triggered = True
                    break
        
        assert action_triggered is True

    def test_hold_detection(self, detector):
         # Avançar período de graça
        landmarks_open = self.create_mock_landmarks(detector, eyes_open=True)
        for _ in range(detector.GRACE_PERIOD_FRAMES + 5):
            detector.process(landmarks_open, 100, 100)
            
        landmarks_closed = self.create_mock_landmarks(detector, eyes_open=False)
        
        # Segurar fechado por muitos frames
        # HOLD_DURATION_SEC = 1.5, assume 30fps -> 45 frames
        # Código usa: if self.left_closed_frames > (30 * HOLD_DURATION_SEC):
        
        # Adicionar frames extras para compensar smoothing (5 frames)
        frames_needed = int(30 * 1.5) + 2 + 5
        
        hold_started = False
        for _ in range(frames_needed):
            res = detector.process(landmarks_closed, 100, 100)
            if res[3]: # hold_start
                hold_started = True
                
        assert hold_started is True
        assert detector.is_holding is True
        
        # Soltar -> deve disparar hold_end
        # Precisa de frames para smoothing voltar a ser aberto
        hold_ended = False
        for _ in range(10):
             res = detector.process(landmarks_open, 100, 100)
             if res[4]: # hold_end
                 hold_ended = True
                 
        assert hold_ended is True

    def test_reflex_detection(self, detector):
        """Testa se piscadas muito rápidas são ignoradas como reflexo."""
        # Avançar período de graça
        landmarks_open = self.create_mock_landmarks(detector, eyes_open=True)
        for _ in range(detector.GRACE_PERIOD_FRAMES + 5):
            detector.process(landmarks_open, 100, 100)
            
        # Simular piscada normal (apenas esquerdo)
        landmarks_closed = self.create_mock_landmarks(detector, left_open=False, right_open=True)
        
        # Fechar (alguns frames)
        for _ in range(5):
            detector.process(landmarks_closed, 100, 100)
            
        assert detector.left_closed_frames > 0
        
        # Abrir MUITO rápido (simulado manipulando histórico ou derivada)
        # Vamos forçar um cenário onde a derivada excede o limite.
        # Mockar BLINK_REFLEX_OPEN_SPEED para ser bem baixo.
        
        with patch('blink_detector.BLINK_REFLEX_OPEN_SPEED', 0.001):
             # Processar abertura
             # Precisamos de alguns frames para a média subir e calcular a derivada
             
             action_triggered = False
             for _ in range(10):
                 res = detector.process(landmarks_open, 100, 100)
                 if res[0]: # left_action
                     action_triggered = True
             
             assert action_triggered is False

    def test_right_click(self, detector):
        """Testa clique com olho direito."""
        # Avançar período de graça
        landmarks_open = self.create_mock_landmarks(detector, eyes_open=True)
        for _ in range(detector.GRACE_PERIOD_FRAMES + 5):
            detector.process(landmarks_open, 100, 100)
            
        # Fechar olho direito
        landmarks_right_closed = self.create_mock_landmarks(detector, left_open=True, right_open=False)
        
        for _ in range(5):
            detector.process(landmarks_right_closed, 100, 100)
            
        assert detector.right_closed_frames > 0
        
        # Abrir e verificar ação
        detector.last_right_blink_time = 0 # Reset cooldown
        
        action_triggered = False
        with patch('blink_detector.BLINK_REFLEX_OPEN_SPEED', 1.0): # Evitar reflexo
            for _ in range(10):
                res = detector.process(landmarks_open, 100, 100)
                if res[1]: # right_action
                    action_triggered = True
                    break
                    
        assert action_triggered is True

    def test_double_blink(self, detector):
        """Testa clique duplo (piscada simultânea)."""
        # Avançar período de graça
        landmarks_open = self.create_mock_landmarks(detector, eyes_open=True)
        for _ in range(detector.GRACE_PERIOD_FRAMES + 5):
            detector.process(landmarks_open, 100, 100)
            
        # Fechar AMBOS os olhos
        landmarks_closed = self.create_mock_landmarks(detector, left_open=False, right_open=False)
        
        for _ in range(5):
            detector.process(landmarks_closed, 100, 100)
            
        assert detector.left_closed_frames > 0
        assert detector.right_closed_frames > 0
        
        # Abrir e verificar ação dupla
        detector.last_left_blink_time = 0
        detector.last_right_blink_time = 0
        
        double_blink_triggered = False
        left_action_triggered = False
        right_action_triggered = False
        
        with patch('blink_detector.BLINK_REFLEX_OPEN_SPEED', 1.0):
            for _ in range(10):
                res = detector.process(landmarks_open, 100, 100)
                # (left, right, double, ...)
                if res[0]: left_action_triggered = True
                if res[1]: right_action_triggered = True
                if res[2]: double_blink_triggered = True
                
        # Se for duplo, left e right devem ser False no retorno final (consumidos)
        assert double_blink_triggered is True
        assert left_action_triggered is False
        assert right_action_triggered is False
