import time
import math
import numpy as np
from config import (
    BLINK_EAR_THRESHOLD,
    BLINK_MIN_FRAMES,
    BLINK_MAX_FRAMES,
    BLINK_COOLDOWN_SEC,
    HOLD_DURATION_SEC
)

class BlinkDetector:
    def __init__(self):
        # Indices dos landmarks para cálculo do EAR (MediaPipe Face Mesh)
        # P1, P2, P3, P4, P5, P6
        self.LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        
        # Estado atual
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        self.left_blink_detected = False
        self.right_blink_detected = False
        self.last_blink_time = 0
        self.hold_start_time = 0
        self.is_holding = False

    def calculate_ear(self, landmarks, indices, img_w, img_h):
        """Calcula o Eye Aspect Ratio (EAR) para um olho."""
        # Extrair coordenadas dos pontos
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append(np.array([lm.x * img_w, lm.y * img_h]))
            
        p1, p2, p3, p4, p5, p6 = coords
        
        # Calcular distâncias verticais
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Calcular distância horizontal
        h = np.linalg.norm(p1 - p4)
        
        # Evitar divisão por zero
        if h == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def process(self, landmarks, img_w, img_h):
        """
        Processa os landmarks e retorna eventos de piscada.
        Retorna: (left_blink, right_blink, double_blink, hold_start, hold_end)
        """
        current_time = time.time()
        
        # Se estiver em cooldown, ignora (exceto para soltar o hold)
        if current_time - self.last_blink_time < BLINK_COOLDOWN_SEC and not self.is_holding:
             return False, False, False, False, False

        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_IDXS, img_w, img_h)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_IDXS, img_w, img_h)
        
        left_closed = left_ear < BLINK_EAR_THRESHOLD
        right_closed = right_ear < BLINK_EAR_THRESHOLD
        
        # --- Lógica Olho Esquerdo ---
        left_action = False
        hold_start = False
        hold_end = False
        
        if left_closed:
            self.left_closed_frames += 1
            
            # Verificar início de hold (arrastar)
            if self.left_closed_frames > (30 * HOLD_DURATION_SEC): # Aprox 30 FPS
                 if not self.is_holding:
                     self.is_holding = True
                     hold_start = True
                     # print("HOLD START")
        else:
            # O olho abriu
            if self.left_closed_frames > 0:
                if self.is_holding:
                    self.is_holding = False
                    hold_end = True
                    # print("HOLD END")
                elif BLINK_MIN_FRAMES <= self.left_closed_frames <= BLINK_MAX_FRAMES:
                    left_action = True
            
            self.left_closed_frames = 0

        # --- Lógica Olho Direito ---
        right_action = False
        if right_closed:
            self.right_closed_frames += 1
        else:
            if self.right_closed_frames > 0:
                 if BLINK_MIN_FRAMES <= self.right_closed_frames <= BLINK_MAX_FRAMES:
                     right_action = True
            self.right_closed_frames = 0
            
        # Detectar piscada simultânea (duplo clique)
        # Simplificação: se ambos detectaram ação no mesmo frame ou muito próximos
        # Na prática, como processamos frame a frame, se ambos fecharam e abriram juntos
        # vamos considerar duplo clique se ambos dispararem 'action' neste frame
        
        double_blink = False
        if left_action and right_action:
            double_blink = True
            left_action = False
            right_action = False
        
        if left_action or right_action or double_blink:
            self.last_blink_time = current_time
            
        return left_action, right_action, double_blink, hold_start, hold_end, (left_ear, right_ear)
