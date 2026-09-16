"""
tracking_validator.py — Validação temporal e estabilização de resultados de rastreamento.

Garante que:
  - Observações antigas ou com timestamp expirado não controlem o cursor.
  - O retorno após perda de rastreamento passe por um período de estabilização
    antes de permitir novos cliques ou gestos, prevenindo falsos disparos.
  - Resultados inválidos sejam explicitamente sinalizados com motivo.
"""
from __future__ import annotations

import logging
import time
from typing import Optional
from frame_data import TrackingResult

logger = logging.getLogger(__name__)


class TrackingValidator:
    """
    Validador temporal e filtro de estabilização para o pipeline.
    """

    def __init__(
        self,
        max_observation_age_sec: float = 0.150,
        stabilization_frames: int = 5,
        stabilization_time_sec: float = 0.200,
    ):
        """
        Args:
            max_observation_age_sec: Idade máxima permitida para um frame gerar ação (s).
            stabilization_frames: Mínimo de frames válidos consecutivos após recuperação.
            stabilization_time_sec: Mínimo de tempo contínuo de rastreamento após recuperação.
        """
        self.max_observation_age_sec = max_observation_age_sec
        self.stabilization_frames = stabilization_frames
        self.stabilization_time_sec = stabilization_time_sec

        self._consecutive_valid_frames: int = 0
        self._tracking_recovered_time: Optional[float] = None
        self._last_validated_frame_id: int = -1
        self._last_valid_result: Optional[TrackingResult] = None

    def validate(self, result: TrackingResult) -> TrackingResult:
        """
        Avalia a validade temporal e o estado de estabilização do resultado.

        Retorna uma nova instância de TrackingResult com tracking_valid e is_stabilized
        atualizados conforme as regras estritas.
        """
        # 1. Checagem de ID estritamente crescente (não reutilizar frames antigos)
        if result.frame_id <= self._last_validated_frame_id and self._last_validated_frame_id != -1:
            logger.warning(
                "Frame descartado: frame_id=%d não é estritamente maior que último validado (%d)",
                result.frame_id, self._last_validated_frame_id,
            )
            return TrackingResult.invalid(
                frame_id=result.frame_id,
                capture_ts=result.capture_timestamp,
                reason="Frame id repetido ou fora de ordem",
            )

        self._last_validated_frame_id = result.frame_id

        # 2. Checagem de validade intrínseca (rosto detectado pelo modelo)
        if not result.tracking_valid or result.landmarks is None:
            self._consecutive_valid_frames = 0
            self._tracking_recovered_time = None
            return TrackingResult(
                frame_id=result.frame_id,
                capture_timestamp=result.capture_timestamp,
                processed_timestamp=result.processed_timestamp,
                landmarks=None,
                features=result.features,
                tracking_valid=False,
                tracking_quality=0.0,
                is_stabilized=False,
                error_message=result.error_message or "Rosto não detectado",
            )

        # 3. Checagem de expiração temporal da observação
        if result.is_expired(self.max_observation_age_sec):
            logger.debug(
                "Observação expirada: idade=%.3fs > limite=%.3fs (frame %d)",
                result.age_sec, self.max_observation_age_sec, result.frame_id,
            )
            return TrackingResult(
                frame_id=result.frame_id,
                capture_timestamp=result.capture_timestamp,
                processed_timestamp=result.processed_timestamp,
                landmarks=result.landmarks,
                features=result.features,
                tracking_valid=False,
                tracking_quality=result.tracking_quality,
                is_stabilized=False,
                error_message=f"Frame expirado ({result.age_sec*1000:.1f}ms)",
            )

        # 4. Controle do período de estabilização pós-recuperação
        now = time.perf_counter()
        if self._tracking_recovered_time is None:
            self._tracking_recovered_time = now
            self._consecutive_valid_frames = 1
        else:
            self._consecutive_valid_frames += 1

        elapsed_since_recovery = now - self._tracking_recovered_time
        has_enough_frames = self._consecutive_valid_frames >= self.stabilization_frames
        has_enough_time = elapsed_since_recovery >= self.stabilization_time_sec

        is_stabilized = has_enough_frames and has_enough_time

        validated_result = TrackingResult(
            frame_id=result.frame_id,
            capture_timestamp=result.capture_timestamp,
            processed_timestamp=result.processed_timestamp,
            landmarks=result.landmarks,
            features=result.features,
            tracking_valid=True,
            tracking_quality=result.tracking_quality,
            is_stabilized=is_stabilized,
            error_message=None if is_stabilized else "Estabilizando rastreamento",
        )

        if is_stabilized:
            self._last_valid_result = validated_result

        return validated_result

    def reset(self) -> None:
        """Reinicia o validador ao trocar de câmera ou calibrar."""
        self._consecutive_valid_frames = 0
        self._tracking_recovered_time = None
        self._last_validated_frame_id = -1
        self._last_valid_result = None
