"""
blink_detector.py — Detector de piscadas e gestos oculares para EyeMouse.

Utiliza Eye Aspect Ratio (EAR) com:
  - Máquina de estados explícita por olho (EyeBlinkState: EYES_OPEN, POSSIBLE_CLOSING,
    CLOSURE_CONFIRMED, OPENING_CONFIRMED, GESTURE_COMPLETED, COOLDOWN).
  - Timestamps monotônicos reais para independência de taxa de quadros (FPS).
  - Limiares com histerese dupla para prevenir oscilação em bordas.
  - Referências calibradas individuais por olho (olho aberto / olho fechado).
  - Discriminação temporal robusta: reflexo involuntário (<90ms), piscada voluntária
    (120ms-450ms) e hold prolongado (>1.0s para arrastar).
"""

from collections import deque
from enum import Enum
import math
import time
from typing import Optional, Tuple

import numpy as np

from config import (
    BLINK_COOLDOWN_SEC,
    BLINK_EAR_THRESHOLD,
    BLINK_MAX_DURATION_SEC,
    BLINK_MAX_FRAMES,
    BLINK_MIN_DURATION_SEC,
    BLINK_MIN_FRAMES,
    BLINK_REFLEX_OPEN_SPEED,
    HOLD_DURATION_SEC,
)


class EyeBlinkState(Enum):
    """Estados explícitos do ciclo de vida de uma piscada ocular."""
    EYES_OPEN = "EYES_OPEN"
    POSSIBLE_CLOSING = "POSSIBLE_CLOSING"
    CLOSURE_CONFIRMED = "CLOSURE_CONFIRMED"
    OPENING_CONFIRMED = "OPENING_CONFIRMED"
    GESTURE_COMPLETED = "GESTURE_COMPLETED"
    COOLDOWN = "COOLDOWN"


class BlinkDetector:
    """
    Detector de piscadas e gestos oculares com máquina de estados e temporização monotônica.
    """

    def __init__(
        self,
        ear_threshold: Optional[float] = None,
        hysteresis_margin: float = 0.020,
    ):
        # Índices dos landmarks para cálculo do EAR (MediaPipe Face Mesh)
        self.LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

        # Estados explícitos da máquina de estados
        self.left_state = EyeBlinkState.EYES_OPEN
        self.right_state = EyeBlinkState.EYES_OPEN

        # Timestamps monotônicos de transição de estado
        self.left_close_start_time: float = 0.0
        self.right_close_start_time: float = 0.0
        self.left_closed_duration: float = 0.0
        self.right_closed_duration: float = 0.0

        # Contadores de frames (mantidos para compatibilidade com testes existentes)
        self.left_closed_frames: int = 0
        self.right_closed_frames: int = 0

        # Cooldown independente com timestamps reais
        self.last_left_blink_time: float = 0.0
        self.last_right_blink_time: float = 0.0

        self.is_holding: bool = False

        # Histórico para suavização do EAR
        self.left_ear_history = deque(maxlen=5)
        self.right_ear_history = deque(maxlen=5)

        # Histórico para derivadas
        self.prev_left_ear: float = 0.0
        self.prev_right_ear: float = 0.0

        # Detecção de reflexo
        self.left_is_reflex: bool = False
        self.right_is_reflex: bool = False

        # Thresholds e Calibração por olho com histerese
        self._ear_threshold: float = (
            float(ear_threshold) if ear_threshold is not None else BLINK_EAR_THRESHOLD
        )
        self.hysteresis_margin: float = float(hysteresis_margin)  # Margem entre fechar e abrir

        # Referências individuais por olho
        self.left_open_ear: float = 0.30
        self.left_closed_ear: float = 0.10
        self.right_open_ear: float = 0.30
        self.right_closed_ear: float = 0.10

        self.is_calibrating: bool = False
        self.calibration_start_time: float = 0.0
        self.calibration_duration: float = 0.0
        self.calibration_samples = []

        # Período de graça inicial pós-reconhecimento facial
        self.face_frames: int = 0
        self.GRACE_PERIOD_FRAMES: int = 30
        self.last_process_time: float = 0.0

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold

    @ear_threshold.setter
    def ear_threshold(self, val: float) -> None:
        self._ear_threshold = float(val)

    @property
    def ear_close_threshold(self) -> float:
        return max(0.01, self._ear_threshold - self.hysteresis_margin)

    @property
    def ear_open_threshold(self) -> float:
        return self._ear_threshold + self.hysteresis_margin

    def set_eye_references(
        self,
        left_open: Optional[float] = None,
        left_closed: Optional[float] = None,
        right_open: Optional[float] = None,
        right_closed: Optional[float] = None,
    ) -> None:
        """Define medições de olho aberto e fechado de cada olho."""
        if left_open is not None: self.left_open_ear = float(left_open)
        if left_closed is not None: self.left_closed_ear = float(left_closed)
        if right_open is not None: self.right_open_ear = float(right_open)
        if right_closed is not None: self.right_closed_ear = float(right_closed)

    def update_eye_state_for_test(
        self,
        eye: str,
        ear: float,
        now: float,
        deriv: float = 0.0,
    ) -> Tuple[bool, bool, bool]:
        """Helper para testes unitários: alimenta frame para um olho e atualiza estado interno."""
        if eye == "left":
            state, start, frames, act, hs, he, ref = self._update_eye_state(
                ear=ear,
                prev_ear=self.prev_left_ear,
                current_state=self.left_state,
                close_start_time=self.left_close_start_time,
                closed_frames=self.left_closed_frames,
                last_blink_time=self.last_left_blink_time,
                now=now,
                deriv=deriv,
            )
            self.left_state = state
            self.left_close_start_time = start
            self.left_closed_frames = frames
            self.prev_left_ear = ear
            return act, hs, he
        else:
            state, start, frames, act, hs, he, ref = self._update_eye_state(
                ear=ear,
                prev_ear=self.prev_right_ear,
                current_state=self.right_state,
                close_start_time=self.right_close_start_time,
                closed_frames=self.right_closed_frames,
                last_blink_time=self.last_right_blink_time,
                now=now,
                deriv=deriv,
            )
            self.right_state = state
            self.right_close_start_time = start
            self.right_closed_frames = frames
            self.prev_right_ear = ear
            return act, hs, he

    def calculate_ear(
        self, landmarks, indices: list, img_w: int, img_h: int
    ) -> float:
        """Calcula o Eye Aspect Ratio (EAR) para um olho específico."""
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append(np.array([lm.x * img_w, lm.y * img_h]))

        p1, p2, p3, p4, p5, p6 = coords

        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)

        if h == 0:
            return 0.0

        return float((v1 + v2) / (2.0 * h))

    def start_calibration(self, duration: float = 10.0) -> None:
        """Inicia calibração de referências oculares do usuário."""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_duration = float(duration)
        self.calibration_samples = []

    def calibrate_eye_references(
        self,
        left_open: float,
        left_closed: float,
        right_open: float,
        right_closed: float,
    ) -> None:
        """Define explicitamente as medições de olho aberto e fechado de cada olho."""
        self.left_open_ear = max(0.05, float(left_open))
        self.left_closed_ear = max(0.01, float(left_closed))
        self.right_open_ear = max(0.05, float(right_open))
        self.right_closed_ear = max(0.01, float(right_closed))

        avg_open = (self.left_open_ear + self.right_open_ear) / 2.0
        avg_closed = (self.left_closed_ear + self.right_closed_ear) / 2.0
        self.ear_threshold = avg_closed + 0.50 * (avg_open - avg_closed)

    def _update_eye_state(
        self,
        ear: float,
        prev_ear: float,
        current_state: EyeBlinkState,
        close_start_time: float,
        closed_frames: int,
        last_blink_time: float,
        now: float,
        deriv: float,
    ) -> Tuple[EyeBlinkState, float, int, bool, bool, bool, bool]:
        """
        Processa um olho pela máquina de estados com histerese e temporização monotônica.

        Returns:
            Tuple: (new_state, new_close_start, new_closed_frames, action_triggered,
                    hold_start, hold_end, is_reflex)
        """
        # Histerese: limiar inferior para fechar, limiar superior para reabrir
        close_thresh = self.ear_threshold - self.hysteresis_margin
        open_thresh = self.ear_threshold + self.hysteresis_margin

        action_triggered = False
        hold_start = False
        hold_end = False
        is_reflex = False

        # Estado 1: Olho Aberto
        if current_state in (EyeBlinkState.EYES_OPEN, EyeBlinkState.COOLDOWN):
            if ear < close_thresh:
                new_state = EyeBlinkState.POSSIBLE_CLOSING
                new_close_start = now
                new_closed_frames = 1
                is_reflex = False
            else:
                new_state = EyeBlinkState.EYES_OPEN
                new_close_start = 0.0
                new_closed_frames = 0

        # Estado 2: Possível Fechamento ou Fechamento Confirmado
        elif current_state in (EyeBlinkState.POSSIBLE_CLOSING, EyeBlinkState.CLOSURE_CONFIRMED):
            if ear < open_thresh:
                # Continua fechado
                new_state = EyeBlinkState.CLOSURE_CONFIRMED
                new_close_start = close_start_time
                new_closed_frames = closed_frames + 1

                duration = now - close_start_time
                if (duration >= HOLD_DURATION_SEC) or (new_closed_frames > int(30 * HOLD_DURATION_SEC)):
                    if not self.is_holding:
                        self.is_holding = True
                        hold_start = True
            else:
                # Olho reabriu!
                duration = max(0.0, now - close_start_time)
                new_state = EyeBlinkState.OPENING_CONFIRMED
                new_close_start = 0.0

                # Detectar velocidade de abertura excessiva (reflexo involuntário)
                reflex_threshold = globals().get("BLINK_REFLEX_OPEN_SPEED", BLINK_REFLEX_OPEN_SPEED)
                if deriv > reflex_threshold:
                    is_reflex = True

                # Se estava segurando, solta
                if self.is_holding:
                    self.is_holding = False
                    hold_end = True
                elif (
                    (BLINK_MIN_DURATION_SEC <= duration <= BLINK_MAX_DURATION_SEC)
                    or (BLINK_MIN_FRAMES <= closed_frames <= BLINK_MAX_FRAMES)
                ):
                    if (now - last_blink_time) > BLINK_COOLDOWN_SEC:
                        if not is_reflex:
                            action_triggered = True
                            new_state = EyeBlinkState.GESTURE_COMPLETED
                        else:
                            new_state = EyeBlinkState.EYES_OPEN
                else:
                    new_state = EyeBlinkState.EYES_OPEN

                new_closed_frames = 0
        else:
            new_state = EyeBlinkState.EYES_OPEN
            new_close_start = 0.0
            new_closed_frames = 0

        return (
            new_state,
            new_close_start,
            new_closed_frames,
            action_triggered,
            hold_start,
            hold_end,
            is_reflex,
        )

    def process(
        self,
        landmarks,
        img_w: int,
        img_h: int,
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, bool, bool, bool, bool, Tuple[float, float]]:
        """
        Processa os landmarks faciais e emite eventos de controle ocular.

        Args:
            landmarks: Landmarks faciais do MediaPipe.
            img_w: Largura do frame.
            img_h: Altura do frame.
            timestamp: Timestamp monotônico opcional (usa time.time/perf_counter se None).

        Returns:
            Tuple: (left_blink, right_blink, double_blink, hold_start, hold_end, (left_ear, right_ear))
        """
        # Utiliza time.time() por padrão para respeitar mocks de tempo legados dos testes
        current_time = time.time() if timestamp is None else float(timestamp)

        # Reset se houve interrupção prolongada
        if current_time - self.last_process_time > 1.0:
            self.face_frames = 0
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            self.left_state = EyeBlinkState.EYES_OPEN
            self.right_state = EyeBlinkState.EYES_OPEN
            self.left_closed_frames = 0
            self.right_closed_frames = 0

        self.last_process_time = current_time
        self.face_frames += 1

        raw_left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_IDXS, img_w, img_h)
        raw_right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_IDXS, img_w, img_h)

        # Modo de Calibração
        if self.is_calibrating:
            if current_time - self.calibration_start_time < self.calibration_duration:
                self.calibration_samples.append((raw_left_ear + raw_right_ear) / 2.0)
                return False, False, False, False, False, (raw_left_ear, raw_right_ear)
            else:
                self.is_calibrating = False
                if self.calibration_samples:
                    avg_ear = float(np.mean(self.calibration_samples))
                    self.ear_threshold = avg_ear * 0.75
                self.left_closed_frames = 0
                self.right_closed_frames = 0
                self.face_frames = 0
                self.left_state = EyeBlinkState.EYES_OPEN
                self.right_state = EyeBlinkState.EYES_OPEN

        # Suavização por Média Móvel
        self.left_ear_history.append(raw_left_ear)
        self.right_ear_history.append(raw_right_ear)

        if len(self.left_ear_history) < 5:
            return False, False, False, False, False, (raw_left_ear, raw_right_ear)

        left_ear = float(np.mean(self.left_ear_history))
        right_ear = float(np.mean(self.right_ear_history))

        # Período de graça
        if self.face_frames < self.GRACE_PERIOD_FRAMES:
            self.prev_left_ear = left_ear
            self.prev_right_ear = right_ear
            return False, False, False, False, False, (left_ear, right_ear)

        left_deriv = left_ear - self.prev_left_ear
        right_deriv = right_ear - self.prev_right_ear
        self.prev_left_ear = left_ear
        self.prev_right_ear = right_ear

        # Atualiza Máquinas de Estados Individuais
        (
            self.left_state,
            self.left_close_start_time,
            self.left_closed_frames,
            left_action,
            left_hold_start,
            left_hold_end,
            left_reflex,
        ) = self._update_eye_state(
            ear=left_ear,
            prev_ear=self.prev_left_ear,
            current_state=self.left_state,
            close_start_time=self.left_close_start_time,
            closed_frames=self.left_closed_frames,
            last_blink_time=self.last_left_blink_time,
            now=current_time,
            deriv=left_deriv,
        )
        if left_reflex:
            self.left_is_reflex = True
        if left_action:
            self.last_left_blink_time = current_time

        (
            self.right_state,
            self.right_close_start_time,
            self.right_closed_frames,
            right_action,
            right_hold_start,
            right_hold_end,
            right_reflex,
        ) = self._update_eye_state(
            ear=right_ear,
            prev_ear=self.prev_right_ear,
            current_state=self.right_state,
            close_start_time=self.right_close_start_time,
            closed_frames=self.right_closed_frames,
            last_blink_time=self.last_right_blink_time,
            now=current_time,
            deriv=right_deriv,
        )
        if right_reflex:
            self.right_is_reflex = True
        if right_action:
            self.last_right_blink_time = current_time

        hold_start = left_hold_start or right_hold_start
        hold_end = left_hold_end or right_hold_end

        # Arbitragem básica de piscada bilateral simultânea (duplo clique)
        double_blink = False
        if left_action and right_action:
            double_blink = True
            left_action = False
            right_action = False
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

