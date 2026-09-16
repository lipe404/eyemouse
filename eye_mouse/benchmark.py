"""
benchmark.py — Sistema de medicao de desempenho interno do EyeMouse.

Mede separadamente cada etapa do pipeline de processamento de frames.
Usa time.perf_counter_ns() para resolucao de nanosegundos.

Importante: a latencia medida por software NAO e a latencia real de
ponta a ponta. VideoCapture.read() nao indica o instante exato de
exposicao do sensor da camera (ha buffer na captura V4L2/DirectShow).
Esta medicao reflete o tempo de processamento interno e e util para
identificar gargalos de CPU, nao para calibrar latencia perceptual.

Uso basico:
    profiler = FrameProfiler()
    with profiler.measure("inference"):
        result = model.detect(frame)
    stats = profiler.stats("inference")
    print(stats)  # mean=12.3ms p95=18.1ms

Modo benchmark (sem mover cursor):
    Configure BENCHMARK_MODE = True em config.py.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Maxima de amostras por etapa mantidas em memoria
_MAX_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageStats:
    """Estatisticas de uma etapa do pipeline."""

    name: str
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def __str__(self) -> str:
        if self.count < 2:
            return f"{self.name}: n={self.count} mean={self.mean_ms:.2f}ms"
        return (
            f"{self.name}: n={self.count} "
            f"mean={self.mean_ms:.2f}ms "
            f"med={self.median_ms:.2f}ms "
            f"p95={self.p95_ms:.2f}ms "
            f"p99={self.p99_ms:.2f}ms "
            f"[{self.min_ms:.2f}…{self.max_ms:.2f}]ms"
        )

    @classmethod
    def insufficient(cls, name: str) -> "StageStats":
        return cls(name=name, count=0, mean_ms=0, median_ms=0,
                   p95_ms=0, p99_ms=0, min_ms=0, max_ms=0)


@dataclass
class PipelineReport:
    """Relatorio completo de todas as etapas para um batch de frames."""

    stages: Dict[str, StageStats]
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    dropped_frames: int = 0
    face_detection_rate: float = 0.0   # Fracao de frames com rosto detectado
    measurement_note: str = (
        "AVISO: latencia de software != latencia perceptual real. "
        "VideoCapture.read() inclui tempo de buffer do driver de camera."
    )

    def __str__(self) -> str:
        lines = [
            f"FPS captura={self.capture_fps:.1f} "
            f"processamento={self.processing_fps:.1f} "
            f"frames_descartados={self.dropped_frames}",
            f"Deteccao de rosto: {self.face_detection_rate*100:.1f}% dos frames",
            "--- Etapas do pipeline ---",
        ]
        for stats in self.stages.values():
            lines.append(f"  {stats}")
        lines.append(self.measurement_note)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timer de contexto
# ---------------------------------------------------------------------------

class _MeasureContext:
    """Gerenciador de contexto para medir uma etapa."""

    __slots__ = ("_profiler", "_name", "_t0")

    def __init__(self, profiler: "FrameProfiler", name: str):
        self._profiler = profiler
        self._name = name
        self._t0: int = 0

    def __enter__(self) -> "_MeasureContext":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ns = time.perf_counter_ns() - self._t0
        self._profiler._record(self._name, elapsed_ns)


# ---------------------------------------------------------------------------
# FrameProfiler
# ---------------------------------------------------------------------------

class FrameProfiler:
    """
    Coleta e agrega medicoes de latencia por etapa do pipeline.

    Thread-safe para leituras concorrentes; cada etapa usa sua propria deque.
    """

    # Nomes canonicos das etapas do pipeline
    STAGE_CAPTURE      = "capture_read"
    STAGE_PREPROCESS   = "preprocess"
    STAGE_INFERENCE    = "inference"
    STAGE_FEATURES     = "feature_extraction"
    STAGE_MAPPING      = "gaze_mapping"
    STAGE_SMOOTHING    = "smoothing"
    STAGE_MOUSE        = "mouse_event"
    STAGE_TOTAL        = "frame_total"

    ALL_STAGES = (
        STAGE_CAPTURE,
        STAGE_PREPROCESS,
        STAGE_INFERENCE,
        STAGE_FEATURES,
        STAGE_MAPPING,
        STAGE_SMOOTHING,
        STAGE_MOUSE,
        STAGE_TOTAL,
    )

    def __init__(self, max_samples: int = _MAX_SAMPLES):
        self._samples: Dict[str, deque] = {
            s: deque(maxlen=max_samples) for s in self.ALL_STAGES
        }
        # Contadores de taxa de frames
        self._capture_times: deque = deque(maxlen=120)
        self._process_times: deque = deque(maxlen=120)
        self._dropped = 0
        self._face_detected_count = 0
        self._total_frames = 0

    # ------------------------------------------------------------------
    # API de medicao
    # ------------------------------------------------------------------

    def measure(self, stage: str) -> _MeasureContext:
        """
        Retorna um gerenciador de contexto para medir uma etapa.

        Exemplo:
            with profiler.measure(FrameProfiler.STAGE_INFERENCE):
                result = detector.detect(frame)
        """
        if stage not in self._samples:
            self._samples[stage] = deque(maxlen=_MAX_SAMPLES)
        return _MeasureContext(self, stage)

    def record_capture_fps(self) -> None:
        """Registra o timestamp de captura de um frame (para calculo de FPS)."""
        self._capture_times.append(time.perf_counter_ns())

    def record_process_fps(self, face_detected: bool = False) -> None:
        """Registra o timestamp de processamento completo de um frame."""
        self._process_times.append(time.perf_counter_ns())
        self._total_frames += 1
        if face_detected:
            self._face_detected_count += 1

    def record_dropped_frame(self) -> None:
        """Incrementa contador de frames descartados da fila."""
        self._dropped += 1

    def record_observation_age(self, age_sec: float) -> None:
        """Registra a idade (segundos) da ultima observacao valida de gaze."""
        self._record("observation_age_ms", int(age_sec * 1e9))

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def stats(self, stage: str) -> StageStats:
        """
        Retorna estatisticas para uma etapa.

        Calcula: media, mediana, p95, p99, min, max.
        Requer >= 2 amostras para percentis; caso contrario retorna zeros.
        """
        samples_ns = list(self._samples.get(stage, []))
        count = len(samples_ns)
        if count == 0:
            return StageStats.insufficient(stage)

        samples_ms = [s / 1e6 for s in samples_ns]
        sorted_ms = sorted(samples_ms)

        mean = sum(sorted_ms) / count
        median = _percentile(sorted_ms, 50)
        p95 = _percentile(sorted_ms, 95)
        p99 = _percentile(sorted_ms, 99)

        return StageStats(
            name=stage,
            count=count,
            mean_ms=mean,
            median_ms=median,
            p95_ms=p95,
            p99_ms=p99,
            min_ms=sorted_ms[0],
            max_ms=sorted_ms[-1],
        )

    @property
    def capture_fps(self) -> float:
        """FPS de captura estimado a partir dos ultimos 120 frames."""
        return _fps_from_timestamps(self._capture_times)

    @property
    def processing_fps(self) -> float:
        """FPS de processamento efetivo."""
        return _fps_from_timestamps(self._process_times)

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    @property
    def face_detection_rate(self) -> float:
        """Fracao de frames onde o rosto foi detectado."""
        if self._total_frames == 0:
            return 0.0
        return self._face_detected_count / self._total_frames

    def report(self) -> PipelineReport:
        """Retorna relatorio completo de todas as etapas."""
        stages = {s: self.stats(s) for s in self._samples}
        return PipelineReport(
            stages=stages,
            capture_fps=self.capture_fps,
            processing_fps=self.processing_fps,
            dropped_frames=self._dropped,
            face_detection_rate=self.face_detection_rate,
        )

    def reset(self) -> None:
        """Zera todas as amostras coletadas."""
        for dq in self._samples.values():
            dq.clear()
        self._capture_times.clear()
        self._process_times.clear()
        self._dropped = 0
        self._face_detected_count = 0
        self._total_frames = 0

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _record(self, stage: str, elapsed_ns: int) -> None:
        if stage not in self._samples:
            self._samples[stage] = deque(maxlen=_MAX_SAMPLES)
        self._samples[stage].append(elapsed_ns)


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list, p: float) -> float:
    """Interpolacao linear para o percentil p em dados ja ordenados."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100.0) * (n - 1)
    lo = int(k)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def _fps_from_timestamps(ts_deque: deque) -> float:
    """Calcula FPS a partir de uma deque de timestamps em nanosegundos."""
    if len(ts_deque) < 2:
        return 0.0
    elapsed_ns = ts_deque[-1] - ts_deque[0]
    if elapsed_ns <= 0:
        return 0.0
    return (len(ts_deque) - 1) / (elapsed_ns / 1e9)
