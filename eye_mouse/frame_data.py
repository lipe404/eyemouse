"""
frame_data.py — Estruturas de dados para o pipeline de visão computacional.

Define contratos imutáveis e padronizados para frames capturados e resultados
de rastreamento, garantindo rastreabilidade temporal e validação estrita.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """
    Representa um frame único capturado da câmera com metadados temporais.

    Atributos:
        frame_id: Identificador estritamente crescente do frame.
        capture_timestamp: Timestamp monotônico no instante da captura (time.perf_counter).
        image: Array NumPy com imagem BGR.
        camera_metadata: Metadados da câmera (resolução, backend, fps, etc.).
    """
    frame_id: int
    capture_timestamp: float
    image: np.ndarray
    camera_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_sec(self) -> float:
        """Idade do frame em segundos desde o instante da captura."""
        return max(0.0, time.perf_counter() - self.capture_timestamp)

    def is_expired(self, max_age_sec: float) -> bool:
        """Retorna True se o frame for mais antigo que max_age_sec."""
        return self.age_sec > max_age_sec


@dataclass(frozen=True)
class TrackingResult:
    """
    Resultado completo do processamento de visão computacional para um frame.

    Atributos:
        frame_id: ID do frame correspondente.
        capture_timestamp: Timestamp monotônico da captura original.
        processed_timestamp: Timestamp monotônico da conclusão do processamento.
        landmarks: Landmarks faciais brutos (ou None se não detectado).
        features: Dicionário com features extraídas (íris, EAR, etc.).
        tracking_valid: True se o rosto e íris foram detectados com confiança.
        tracking_quality: Score de qualidade do rastreamento (0.0 a 1.0).
        is_stabilized: True se o período de estabilização pós-recuperação foi concluído.
        error_message: Mensagem descritiva em caso de falha ou invalidação.
    """
    frame_id: int
    capture_timestamp: float
    processed_timestamp: float
    landmarks: Any = None
    features: Dict[str, Any] = field(default_factory=dict)
    tracking_valid: bool = False
    tracking_quality: float = 0.0
    is_stabilized: bool = False
    error_message: Optional[str] = None

    @property
    def latency_sec(self) -> float:
        """Latência do processamento (da captura até a geração do resultado)."""
        return max(0.0, self.processed_timestamp - self.capture_timestamp)

    @property
    def age_sec(self) -> float:
        """Idade total da observação desde a captura até o instante atual."""
        return max(0.0, time.perf_counter() - self.capture_timestamp)

    def is_expired(self, max_age_sec: float) -> bool:
        """Retorna True se a observação ultrapassar a validade temporal máxima."""
        return self.age_sec > max_age_sec

    @classmethod
    def invalid(cls, frame_id: int, capture_ts: float, reason: str) -> "TrackingResult":
        """Construtor de conveniência para resultado sem rastreamento válido."""
        return cls(
            frame_id=frame_id,
            capture_timestamp=capture_ts,
            processed_timestamp=time.perf_counter(),
            landmarks=None,
            features={},
            tracking_valid=False,
            tracking_quality=0.0,
            is_stabilized=False,
            error_message=reason,
        )
