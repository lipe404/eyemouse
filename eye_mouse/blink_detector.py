import time
import math
import numpy as np
from collections import deque
from config import (
    BLINK_EAR_THRESHOLD,
    BLINK_MIN_FRAMES,
    BLINK_MAX_FRAMES,
    BLINK_COOLDOWN_SEC,
    HOLD_DURATION_SEC,
    BLINK_REFLEX_OPEN_SPEED
)


class BlinkDetector:
    """
    Detecta piscadas e gestos oculares para controle do mouse.

    Utiliza o Eye Aspect Ratio (EAR) para determinar se os olhos estão abertos
    ou fechados. Implementa lógica para diferenciar piscadas voluntárias (cliques)
    de reflexos involuntários, além de permitir calibração dinâmica do threshold.
    """

    def __init__(self):
        """Inicializa o detector de piscadas com parâmetros padrão."""
        # Indices dos landmarks para cálculo do EAR (MediaPipe Face Mesh)
        self.LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

        # Estado atual
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        
        # Cooldown independente (Melhoria 8)
        self.last_left_blink_time = 0
        self.last_right_blink_time = 0
        
        self.is_holding = False

        # Histórico para suavização do EAR (Aumentado para 5 - Melhoria 11)
        self.left_ear_history = deque(maxlen=5)
        self.right_ear_history = deque(maxlen=5)
        
        # Histórico anterior para derivada
        self.prev_left_ear = 0.0
        self.prev_right_ear = 0.0
        
        # Detecção de reflexo (Melhoria 7)
        self.left_is_reflex = False
        self.right_is_reflex = False
        
        # Threshold dinâmico (Melhoria 6)
        self.ear_threshold = BLINK_EAR_THRESHOLD
        self.is_calibrating = False
        self.calibration_start_time = 0
        self.calibration_duration = 0
        self.calibration_samples = []
        
        # Período de graça (Melhoria 9)
        self.face_frames = 0
        self.GRACE_PERIOD_FRAMES = 30
        self.last_process_time = 0

    def calculate_ear(self, landmarks, indices, img_w, img_h):
        """
        Calcula o Eye Aspect Ratio (EAR) para um olho.

        O EAR é uma medida da abertura do olho, baseada na relação entre
        a altura e a largura do olho nos landmarks.

        Args:
            landmarks: Lista de landmarks faciais do MediaPipe.
            indices (list): Índices dos landmarks correspondentes ao olho.
            img_w (int): Largura da imagem em pixels.
            img_h (int): Altura da imagem em pixels.

        Returns:
            float: O valor do EAR calculado. Retorna 0.0 se a largura for 0.
        """
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

    def start_calibration(self, duration=10.0):
        """
        Inicia a calibração do threshold de piscada.

        Durante a calibração, o sistema coleta amostras de EAR para definir
        um threshold personalizado baseado na fisionomia do usuário.

        Args:
            duration (float): Duração da calibração em segundos.
        """
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_duration = duration
        self.calibration_samples = []
        print(f"Iniciando calibração de piscada por {duration}s...")

    def process(self, landmarks, img_w, img_h):
        """
        Processa os landmarks e retorna eventos de piscada.

        Analisa o estado dos olhos (aberto/fechado), detecta gestos (piscada,
        piscada longa para arrastar) e filtra reflexos involuntários.

        Args:
            landmarks: Lista de landmarks faciais do MediaPipe.
            img_w (int): Largura da imagem.
            img_h (int): Altura da imagem.

        Returns:
            tuple: (left_blink, right_blink, double_blink, hold_start, hold_end, ears)
                - left_blink (bool): True se houve clique esquerdo.
                - right_blink (bool): True se houve clique direito.
                - double_blink (bool): True se houve clique duplo (não implementado ainda).
                - hold_start (bool): True se iniciou modo arrastar.
                - hold_end (bool): True se finalizou modo arrastar.
                - ears (tuple): (left_ear, right_ear) atuais.
        """
        current_time = time.time()
        
        # Resetar contador de frames se houve gap grande (rosto saiu e voltou)
        if current_time - self.last_process_time > 1.0:
            self.face_frames = 0
            # Limpar históricos para evitar picos falsos
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            
        self.last_process_time = current_time
        self.face_frames += 1

        # Calcular EAR bruto
        raw_left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_IDXS, img_w, img_h)
        raw_right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_IDXS, img_w, img_h)

        # Se estiver calibrando, apenas coleta dados
        if self.is_calibrating:
            if current_time - self.calibration_start_time < self.calibration_duration:
                self.calibration_samples.append((raw_left_ear + raw_right_ear) / 2.0)
                # Retorna tudo falso durante calibração
                return False, False, False, False, False, (raw_left_ear, raw_right_ear)
            else:
                # Fim da calibração
                self.is_calibrating = False
                if self.calibration_samples:
                    avg_ear = np.mean(self.calibration_samples)
                    # Define threshold como 75% da média (margem segura)
                    # Se olho aberto é 0.30 -> threshold 0.225
                    self.ear_threshold = avg_ear * 0.75
                    print(f"Calibração concluída. Novo Threshold EAR: {self.ear_threshold:.3f} (Média: {avg_ear:.3f})")
                
                # Resetar estados para evitar clique pós-calibração
                self.left_closed_frames = 0
                self.right_closed_frames = 0
                self.face_frames = 0 

        # Suavização (Média Móvel)
        self.left_ear_history.append(raw_left_ear)
        self.right_ear_history.append(raw_right_ear)

        # Precisamos de buffer cheio para estabilidade
        if len(self.left_ear_history) < 5:
             return False, False, False, False, False, (raw_left_ear, raw_right_ear)

        left_ear = np.mean(self.left_ear_history)
        right_ear = np.mean(self.right_ear_history)
        
        # Ignorar piscadas durante o período de graça (Melhoria 9)
        if self.face_frames < self.GRACE_PERIOD_FRAMES:
             # Atualizar prev_ear mesmo ignorando para derivada funcionar depois
             self.prev_left_ear = left_ear
             self.prev_right_ear = right_ear
             return False, False, False, False, False, (left_ear, right_ear)

        # Calcular derivadas (velocidade de mudança)
        left_deriv = left_ear - self.prev_left_ear
        right_deriv = right_ear - self.prev_right_ear
        
        self.prev_left_ear = left_ear
        self.prev_right_ear = right_ear

        # Detectar estado fechado usando threshold dinâmico
        left_closed = left_ear < self.ear_threshold
        right_closed = right_ear < self.ear_threshold

        # --- Lógica Olho Esquerdo (Clique Esquerdo / Arrastar) ---
        left_action = False
        hold_start = False
        hold_end = False

        if left_closed:
            self.left_closed_frames += 1
            # Resetar flag de reflexo ao começar a fechar
            if self.left_closed_frames == 1:
                self.left_is_reflex = False

            # Verificar início de hold (arrastar)
            if self.left_closed_frames > (30 * HOLD_DURATION_SEC):
                if not self.is_holding:
                    self.is_holding = True
                    hold_start = True
        else:
            # O olho está aberto ou abrindo
            if self.left_closed_frames > 0:
                # Detectar abertura muito rápida (reflexo)
                if left_deriv > BLINK_REFLEX_OPEN_SPEED:
                    self.left_is_reflex = True

                # Se estava segurando, solta (independente de reflexo, segurança)
                if self.is_holding:
                    self.is_holding = False
                    hold_end = True
                
                # Se foi rápido e NÃO foi reflexo, é clique
                elif BLINK_MIN_FRAMES <= self.left_closed_frames <= BLINK_MAX_FRAMES:
                    # Verifica cooldown independente
                    if current_time - self.last_left_blink_time > BLINK_COOLDOWN_SEC:
                        if not self.left_is_reflex:
                             left_action = True
                             self.last_left_blink_time = current_time
                        else:
                             pass # Ignorado por ser reflexo

            self.left_closed_frames = 0

        # --- Lógica Olho Direito (Clique Direito) ---
        right_action = False

        if right_closed:
            self.right_closed_frames += 1
            if self.right_closed_frames == 1:
                self.right_is_reflex = False
        else:
            if self.right_closed_frames > 0:
                # Detectar abertura muito rápida (reflexo)
                if right_deriv > BLINK_REFLEX_OPEN_SPEED:
                    self.right_is_reflex = True
                
                if BLINK_MIN_FRAMES <= self.right_closed_frames <= BLINK_MAX_FRAMES:
                    if current_time - self.last_right_blink_time > BLINK_COOLDOWN_SEC:
                        if not self.right_is_reflex:
                            right_action = True
                            self.last_right_blink_time = current_time

            self.right_closed_frames = 0

        # Detectar piscada simultânea (duplo clique)
        double_blink = False
        if left_action and right_action:
            double_blink = True
            left_action = False
            right_action = False
            # Atualizar ambos cooldowns
            self.last_left_blink_time = current_time
            self.last_right_blink_time = current_time

        return (
            left_action,
            right_action,
            double_blink,
            hold_start,
            hold_end,
            (left_ear, right_ear),
        )
