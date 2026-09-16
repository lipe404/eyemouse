"""
smoothing.py — Filtros de suavização de cursor intercambiáveis.

Fornece implementações para suavização de trajetórias de olhar/mouse:
  - BaseSmoothingFilter: Interface abstrata para todos os filtros.
  - OneEuroFilter: Filtro 1€ adaptativo à velocidade (Casiez et al., CHI 2012).
    Alta suavidade em fixações (repouso) e baixíssima latência em sacadas (movimentos rápidos).
  - KalmanSmoothingFilter: Filtro baseado em cv2.KalmanFilter (preservado como referência).
    Lookahead fixo de 66ms removido por padrão para evitar overshoot e oscilações.
  - PassThroughFilter: Retorna coordenadas puras sem suavização (baseline de latência zero).
  - SmoothingFilter: Alias de compatibilidade para KalmanSmoothingFilter.
"""

from abc import ABC, abstractmethod
from collections import deque
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class BaseSmoothingFilter(ABC):
    """Interface abstrata para filtros de suavização de cursor."""

    @abstractmethod
    def update(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Processa nova medição bruta (x, y) e retorna a posição filtrada.

        Args:
            x: Coordenada X observada.
            y: Coordenada Y observada.
            timestamp: Timestamp monotônico em segundos (None = usa time.perf_counter).

        Returns:
            Tuple[float, float]: Coordenadas (x, y) suavizadas.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reinicia o estado interno do filtro (ex: após perda de rastreamento)."""
        pass

    def set_alpha(self, alpha: float) -> None:
        """
        Ajuste genérico de sensibilidade (0.0 = máximo suave, 1.0 = rápido/sem filtro).
        Implementações mapeiam esse valor para seus parâmetros internos.
        """
        pass


class _LowPassFilter1D:
    """Filtro passa-baixa de primeira ordem (1D) para formulação do One Euro Filter."""

    def __init__(self, init_value: float = 0.0):
        self._val: float = float(init_value)
        self._last_raw: float = float(init_value)
        self._initialized: bool = False

    def filter(self, value: float, alpha: float) -> float:
        val_f = float(value)
        if not self._initialized:
            self._val = val_f
            self._last_raw = val_f
            self._initialized = True
            return val_f

        self._val = alpha * val_f + (1.0 - alpha) * self._val
        self._last_raw = val_f
        return self._val

    @property
    def last_filtered(self) -> float:
        return self._val

    @property
    def last_raw(self) -> float:
        return self._last_raw

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self._val = 0.0
        self._last_raw = 0.0
        self._initialized = False


def _compute_alpha(cutoff_hz: float, dt_sec: float) -> float:
    """
    Calcula o coeficiente alpha de suavização:
        alpha = 1 / (1 + tau / dt) = (2 * pi * fc * dt) / (1 + 2 * pi * fc * dt)
    onde tau = 1 / (2 * pi * fc).
    """
    if cutoff_hz <= 0.0 or dt_sec <= 0.0:
        return 1.0
    r = 2.0 * math.pi * cutoff_hz * dt_sec
    return max(0.0, min(1.0, r / (r + 1.0)))


class OneEuroFilter(BaseSmoothingFilter):
    """
    Implementação do One Euro Filter (1€ Filter) segundo Casiez et al. (CHI 2012).

    Adapta a frequência de corte (cutoff) com base na velocidade estimada do sinal:
      - Velocidade baixa (fixação do olhar): cutoff se aproxima de `min_cutoff`,
        eliminando micro-jitter e ruído do sensor sem zona morta mecânica.
      - Velocidade alta (sacada): cutoff aumenta proporcionalmente a `beta * |v|`,
        eliminando lag e mantendo resposta instantânea.

    Parâmetros:
        min_cutoff (fc_min): Frequência de corte mínima em Hz (repouso). Padrão: 1.0 Hz.
        beta: Coeficiente de resposta à velocidade. Padrão: 0.007.
        d_cutoff (fc_d): Frequência de corte do filtro da derivada em Hz. Padrão: 1.0 Hz.
        enable_deadzone: Se True, aplica histerese adaptativa para estabilização extra.
        deadzone_rest: Raio em pixels para ativação da zona morta em repouso.
        deadzone_release: Raio em pixels para liberação da zona morta ao iniciar movimento.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        enable_deadzone: bool = False,
        deadzone_rest: float = 2.0,
        deadzone_release: float = 4.5,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.enable_deadzone = enable_deadzone
        self.deadzone_rest = float(deadzone_rest)
        self.deadzone_release = float(deadzone_release)

        # Filtros de coordenada
        self._x_filter = _LowPassFilter1D()
        self._y_filter = _LowPassFilter1D()

        # Filtros da derivada (velocidade)
        self._dx_filter = _LowPassFilter1D()
        self._dy_filter = _LowPassFilter1D()

        self._last_time: Optional[float] = None
        self._in_fixation_deadzone: bool = False
        self._anchor_pos: Optional[Tuple[float, float]] = None

    def update(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        now = time.perf_counter() if timestamp is None else float(timestamp)

        # Primeiro ponto: inicializa e retorna diretamente sem lag
        if self._last_time is None or not self._x_filter.is_initialized:
            self._last_time = now
            self._x_filter.filter(x, 1.0)
            self._y_filter.filter(y, 1.0)
            self._dx_filter.reset()
            self._dy_filter.reset()
            self._anchor_pos = (float(x), float(y))
            return float(x), float(y)

        dt = now - self._last_time
        self._last_time = now

        # Tratamento rigoroso de dt inválido, negativo ou muito longo
        if dt <= 1e-5:
            dt = 1.0 / 30.0  # Fallback defensivo para intervalos nulos/invertidos
        elif dt > 1.0:
            # Pausa muito longa (> 1s): reinicia estado para evitar derivação espúria
            self.reset()
            return self.update(x, y, timestamp=now)

        # 1. Derivada bruta (velocidade Instantânea)
        dx_raw = (x - self._x_filter.last_filtered) / dt
        dy_raw = (y - self._y_filter.last_filtered) / dt

        # 2. Derivada filtrada com d_cutoff
        alpha_d = _compute_alpha(self.d_cutoff, dt)
        dx_hat = self._dx_filter.filter(dx_raw, alpha_d)
        dy_hat = self._dy_filter.filter(dy_raw, alpha_d)

        # 3. Frequência de corte dinâmica adaptada à velocidade
        cutoff_x = self.min_cutoff + self.beta * abs(dx_hat)
        cutoff_y = self.min_cutoff + self.beta * abs(dy_hat)

        # 4. Filtragem da posição
        alpha_x = _compute_alpha(cutoff_x, dt)
        alpha_y = _compute_alpha(cutoff_y, dt)

        filtered_x = self._x_filter.filter(x, alpha_x)
        filtered_y = self._y_filter.filter(y, alpha_y)

        # 5. Histerese e Zona Morta Adaptativa (opcional)
        if self.enable_deadzone:
            filtered_x, filtered_y = self._apply_hysteresis_deadzone(
                filtered_x, filtered_y, abs(dx_hat) + abs(dy_hat)
            )

        return float(filtered_x), float(filtered_y)

    def _apply_hysteresis_deadzone(
        self, x: float, y: float, speed: float
    ) -> Tuple[float, float]:
        """Aplica histerese com raio de repouso e raio de liberação."""
        if self._anchor_pos is None:
            self._anchor_pos = (x, y)
            return x, y

        ax, ay = self._anchor_pos
        dist = math.hypot(x - ax, y - ay)

        if self._in_fixation_deadzone:
            # Em fixação: só rompe a zona morta se exceder o raio de liberação
            if dist > self.deadzone_release:
                self._in_fixation_deadzone = False
                self._anchor_pos = (x, y)
                return x, y
            return ax, ay
        else:
            # Em movimento: se a velocidade caiu e o movimento é minúsculo, entra em fixação
            if dist < self.deadzone_rest and speed < 30.0:
                self._in_fixation_deadzone = True
                self._anchor_pos = (x, y)
                return ax, ay
            self._anchor_pos = (x, y)
            return x, y

    def reset(self) -> None:
        self._x_filter.reset()
        self._y_filter.reset()
        self._dx_filter.reset()
        self._dy_filter.reset()
        self._last_time = None
        self._in_fixation_deadzone = False
        self._anchor_pos = None

    def set_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        """Atualiza parâmetros dinamicamente."""
        if min_cutoff is not None:
            self.min_cutoff = max(0.01, float(min_cutoff))
        if beta is not None:
            self.beta = max(0.0, float(beta))
        if d_cutoff is not None:
            self.d_cutoff = max(0.01, float(d_cutoff))

    def set_alpha(self, alpha: float) -> None:
        """
        Mapeia alpha (0.01 a 1.0) para min_cutoff:
          alpha 0.01 -> min_cutoff 0.2 Hz (máximo suave)
          alpha 1.0  -> min_cutoff 5.0 Hz (alta agilidade)
        """
        alpha_c = max(0.01, min(1.0, float(alpha)))
        self.min_cutoff = 0.2 + alpha_c * 4.8


class KalmanSmoothingFilter(BaseSmoothingFilter):
    """
    Filtro de Kalman para suavização de cursor (preservado como benchmark e referência).

    Estado: [x, y, dx, dy] (posição e velocidade)
    Medição: [x, y] (posição observada)

    Por padrão, `lookahead = 0.0` para eliminar o overshoot e oscilações do modelo anterior.
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 1e-1,
        lookahead: float = 0.0,
    ):
        self.kf = cv2.KalmanFilter(4, 2)

        # Matriz de Medição (H): Observamos apenas x e y
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )

        # Matriz de Transição (F)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )

        self.process_noise_base = float(process_noise)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)

        self.base_measurement_noise = float(measurement_noise)
        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * float(measurement_noise)
        )

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.last_time: Optional[float] = None
        self.first_update: bool = True
        self.prev_output: Optional[Tuple[float, float]] = None

        self.position_buffer = deque(maxlen=5)
        self.dead_zone_rest = 15.0
        self.dead_zone_active = 2.0
        self.current_alpha = 0.5
        self.lookahead = float(lookahead)

    def _sigmoid(self, x: float, midpoint: float = 50, steepness: float = 0.1) -> float:
        """Curva sigmóide para transição suave baseada na velocidade."""
        return float(1.0 / (1.0 + np.exp(-steepness * (x - midpoint))))

    def update(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        current_time = time.time() if timestamp is None else float(timestamp)

        self.position_buffer.append((x, y))
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if self.first_update:
            self.kf.statePost = np.array(
                [[np.float32(x)], [np.float32(y)], [0], [0]], np.float32
            )
            self.last_time = current_time
            self.first_update = False
            self.prev_output = (float(x), float(y))
            return float(x), float(y)

        dt = current_time - (self.last_time or current_time)
        self.last_time = current_time

        if dt <= 0 or dt > 1.0:
            dt = 1.0 / 30.0

        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        vx_est = abs(float(self.kf.statePost[2][0]))
        vy_est = abs(float(self.kf.statePost[3][0]))

        response_x = self._sigmoid(vx_est, midpoint=100, steepness=0.05)
        response_y = self._sigmoid(vy_est, midpoint=100, steepness=0.05)

        base_r = self.base_measurement_noise
        r_x = base_r / (1.0 + 10.0 * response_x)
        r_y = base_r / (1.0 + 10.0 * response_y)

        self.kf.measurementNoiseCov[0, 0] = r_x
        self.kf.measurementNoiseCov[1, 1] = r_y

        prediction = self.kf.predict()
        estimated = self.kf.correct(measurement)

        est_x = float(estimated[0][0])
        est_y = float(estimated[1][0])
        est_vx = float(estimated[2][0])
        est_vy = float(estimated[3][0])

        vel_magnitude = math.hypot(est_vx, est_vy)
        dz_factor = max(0.0, min(1.0, vel_magnitude / 500.0))
        current_dead_zone = (
            self.dead_zone_rest * (1.0 - dz_factor)
            + self.dead_zone_active * dz_factor
        )

        if self.prev_output:
            prev_x, prev_y = self.prev_output
            dist = math.hypot(est_x - prev_x, est_y - prev_y)

            if dist < current_dead_zone:
                est_x, est_y = prev_x, prev_y
                est_vx, est_vy = 0.0, 0.0

        # Aplica lookahead se explicitamente configurado (> 0)
        final_x = est_x + est_vx * self.lookahead
        final_y = est_y + est_vy * self.lookahead

        self.prev_output = (final_x, final_y)
        return float(final_x), float(final_y)

    def reset(self) -> None:
        self.first_update = True
        self.last_time = None
        self.kf.statePost = np.zeros((4, 1), np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.position_buffer.clear()
        self.prev_output = None

    def set_alpha(self, alpha: float) -> None:
        alpha = max(0.01, min(1.0, float(alpha)))
        self.current_alpha = alpha
        exponent = (1.0 - alpha) * 4.0 - 3.0
        self.base_measurement_noise = 10.0 ** exponent


# Alias de retrocompatibilidade: SmoothingFilter aponta para KalmanSmoothingFilter
SmoothingFilter = KalmanSmoothingFilter


class PassThroughFilter(BaseSmoothingFilter):
    """Filtro nulo (Pass-Through): retorna as coordenadas observadas diretamente."""

    def __init__(self):
        self._last_pos: Optional[Tuple[float, float]] = None

    def update(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        self._last_pos = (float(x), float(y))
        return float(x), float(y)

    def reset(self) -> None:
        self._last_pos = None

    def set_alpha(self, alpha: float) -> None:
        pass


def create_smoothing_filter(
    filter_type: str = "ONE_EURO", **kwargs
) -> BaseSmoothingFilter:
    """
    Factory para instanciação de filtros de suavização.

    Args:
        filter_type: "ONE_EURO", "KALMAN", "NONE" ou "PASSTHROUGH".
        **kwargs: Parâmetros específicos passados ao construtor do filtro.

    Returns:
        BaseSmoothingFilter: Instância do filtro configurado.
    """
    f_type = filter_type.upper().strip()
    if f_type in ("ONE_EURO", "1EURO", "ONEEURO"):
        return OneEuroFilter(**kwargs)
    elif f_type in ("KALMAN", "KF"):
        return KalmanSmoothingFilter(**kwargs)
    elif f_type in ("NONE", "PASSTHROUGH", "NO_FILTER"):
        return PassThroughFilter()
    else:
        raise ValueError(
            f"Tipo de filtro desconhecido: '{filter_type}'. "
            "Use 'ONE_EURO', 'KALMAN' ou 'NONE'."
        )
