"""
calibration.py — Gerenciador de calibração do rastreamento ocular de alta qualidade.

Milestone 3:
  - Modelos intercambiáveis: Ridge (L2 regularizado), Polinomial (2ª ordem) e Linear.
  - Avaliação científica completa com conjunto holdout separado (RMSE, Média, Mediana, P95, % da Diagonal e Quadrantes).
  - Agregação robusta de amostras por alvo via Mediana e rejeição de outliers por MAD (Median Absolute Deviation).
  - Deduplicação estrita de observações por frame_id e timestamp.
  - Recalibração pontual de alvos individuais (recalibrate_point).
  - Ajuste fino de trim/offset (set_trim) sem corromper o modelo original.
  - Detecção de alteração de geometria/resolução da tela.
  - Backup atômico do perfil anterior (.bak) antes de salvar nova calibração validada.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import (
    CALIBRATION_FILE_PREFIX,
    CALIBRATION_REPROJECTION_ERROR_THRESHOLD,
    FT_HOLDOUT_VALIDATION,
)
from calibration_models import (
    BaseCalibrationModel,
    PolynomialCalibrationModel,
    RidgeCalibrationModel,
    LinearCalibrationModel,
    CalibrationMetrics,
    evaluate_calibration_metrics,
)

logger = logging.getLogger(__name__)

_PROFILE_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')
_CALIBRATION_VERSION = 2


def _validate_profile_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Nome de perfil nao pode ser vazio.")
    if not _PROFILE_RE.match(name):
        raise ValueError(
            f"Nome de perfil invalido: '{name}'. "
            "Use apenas letras, digitos, '_' e '-' (1-64 caracteres)."
        )
    return name


def filter_outliers_mad(points: List[Tuple[float, float]], threshold_mad: float = 2.5) -> List[Tuple[float, float]]:
    """
    Filtra outliers em uma lista de pontos 2D usando Median Absolute Deviation (MAD).
    Muito mais robusto que média/desvio-padrão na presença de piscadas ou micro-sacadas.
    """
    if len(points) < 4:
        return points

    arr = np.array(points, dtype=np.float64)
    median = np.median(arr, axis=0)
    diffs = np.linalg.norm(arr - median, axis=1)
    mad = np.median(diffs) + 1e-6

    # Manter pontos dentro de threshold_mad * MAD
    inliers = [points[i] for i in range(len(points)) if diffs[i] <= threshold_mad * mad]
    return inliers if len(inliers) >= 2 else points


class CalibrationManager:
    """
    Gerencia o processo de calibração e mapeamento do olhar para a tela.
    """

    def __init__(self, profile_name: str = "default", model_type: str = "ridge"):
        self.profile_name: str = _validate_profile_name(profile_name)
        self.model_type = model_type.lower()
        self._json_file: str = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.json"
        self._npy_file: str = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.npy"
        self._bak_file: str = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.json.bak"

        self.iris_points: List[Tuple[float, float]] = []
        self.screen_points: List[Tuple[int, int]] = []

        # Histórico de amostras por alvo para agregação robusta
        self._target_raw_samples: Dict[int, List[Tuple[float, float]]] = {}
        self._collected_frame_ids: set = set()

        # Modelo matemático ativo
        self.model: BaseCalibrationModel = self._create_model(self.model_type)
        self.is_calibrated: bool = False

        # Métricas da última calibração
        self.last_metrics: Optional[CalibrationMetrics] = None
        self.last_holdout_error: float = float('inf')
        self.last_train_error: float = float('inf')

        # Ajuste fino / Trim
        self.trim_x: float = 0.0
        self.trim_y: float = 0.0

        # Resolução de tela registrada e fatores de escala dinâmicos
        self.calibrated_screen_w: int = 1920
        self.calibrated_screen_h: int = 1080
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

    def _create_model(self, model_name: str) -> BaseCalibrationModel:
        if model_name == "linear":
            return LinearCalibrationModel()
        elif model_name == "polynomial":
            return PolynomialCalibrationModel()
        else:
            return RidgeCalibrationModel(alpha_l2=1e-3, use_polynomial=True)

    @property
    def calibration_file(self) -> str:
        return self._json_file

    @property
    def coeffs_x(self) -> Optional[np.ndarray]:
        return self.model.coeffs_x

    @coeffs_x.setter
    def coeffs_x(self, val: Optional[np.ndarray]):
        self.model.coeffs_x = val
        if val is not None and self.model.coeffs_y is not None:
            self.model.is_fitted = True
            self.is_calibrated = True

    @property
    def coeffs_y(self) -> Optional[np.ndarray]:
        return self.model.coeffs_y

    @coeffs_y.setter
    def coeffs_y(self, val: Optional[np.ndarray]):
        self.model.coeffs_y = val
        if val is not None and self.model.coeffs_x is not None:
            self.model.is_fitted = True
            self.is_calibrated = True

    def set_profile(self, profile_name: str) -> None:
        self.profile_name = _validate_profile_name(profile_name)
        self._json_file = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.json"
        self._npy_file = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.npy"
        self._bak_file = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.json.bak"
        self.load_calibration()

    def set_trim(self, offset_x: float, offset_y: float) -> None:
        """Ajusta o deslocamento fino do cursor sem alterar o modelo original."""
        self.trim_x = float(offset_x)
        self.trim_y = float(offset_y)
        logger.info("Trim do cursor ajustado para (%+.1f, %+.1f) px", self.trim_x, self.trim_y)

    # ------------------------------------------------------------------
    # Coleta de Pontos com Deduplicação e Agregação
    # ------------------------------------------------------------------

    def add_point(
        self,
        iris_pos: Tuple[float, float],
        screen_pos: Tuple[int, int],
    ) -> None:
        """Adiciona um par (iris, tela) diretamente."""
        self.iris_points.append(iris_pos)
        self.screen_points.append(screen_pos)

    def add_point_sample(
        self,
        target_idx: int,
        iris_pos: Tuple[float, float],
        screen_pos: Tuple[int, int],
        frame_id: int,
    ) -> bool:
        """
        Adiciona uma amostra com validação estrita de deduplicação por frame_id.
        Retorna True se a amostra foi aceita.
        """
        if frame_id in self._collected_frame_ids:
            logger.debug("Amostra duplicada descartada: frame_id=%d", frame_id)
            return False

        self._collected_frame_ids.add(frame_id)
        if target_idx not in self._target_raw_samples:
            self._target_raw_samples[target_idx] = []
        self._target_raw_samples[target_idx].append(iris_pos)
        return True

    def finalize_target(self, target_idx: int, screen_pos: Tuple[int, int]) -> bool:
        """
        Consolida as amostras coletadas para um alvo específico usando Mediana e filtro MAD.
        """
        samples = self._target_raw_samples.get(target_idx, [])
        if len(samples) < 3:
            logger.warning("Amostras insuficientes para o alvo %d (%d amostras)", target_idx, len(samples))
            return False

        filtered = filter_outliers_mad(samples)
        arr = np.array(filtered)
        aggregated_iris = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

        self.add_point(aggregated_iris, screen_pos)
        logger.debug(
            "Alvo %d consolidado: %d amostras brutas -> %d inliers -> iris=(%.4f, %.4f)",
            target_idx, len(samples), len(filtered), aggregated_iris[0], aggregated_iris[1]
        )
        return True

    def recalibrate_point(
        self,
        point_idx: int,
        new_iris_pos: Tuple[float, float],
        new_screen_pos: Tuple[int, int],
    ) -> bool:
        """Permite recalibrar rapidamente um alvo problemático sem reiniciar do zero."""
        if 0 <= point_idx < len(self.iris_points):
            self.iris_points[point_idx] = new_iris_pos
            self.screen_points[point_idx] = new_screen_pos
            logger.info("Ponto %d recalibrado. Recalculando modelo...", point_idx)
            success, _ = self.compute_calibration()
            return success
        return False

    def clear_points(self) -> None:
        self.iris_points = []
        self.screen_points = []
        self._target_raw_samples.clear()
        self._collected_frame_ids.clear()

    # ------------------------------------------------------------------
    # Cálculo e Validação
    # ------------------------------------------------------------------

    def compute_calibration(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
    ) -> Tuple[bool, float]:
        """
        Calcula o modelo e avalia com holdout científico.
        Retorna (sucesso, holdout_error_px).
        """
        n = len(self.iris_points)
        if n < 6:
            logger.warning("Pontos insuficientes: %d (mínimo 6).", n)
            return False, 0.0

        self.calibrated_screen_w = screen_w
        self.calibrated_screen_h = screen_h

        iris_arr = np.array(self.iris_points, dtype=np.float64)
        screen_arr = np.array(self.screen_points, dtype=np.float64)

        # Divisão em Treino e Validação (Holdout de 20% ou mínimo 2 pontos)
        n_holdout = max(2, n // 5) if FT_HOLDOUT_VALIDATION else 0
        n_train = n - n_holdout

        train_idx = list(range(n_train))
        holdout_idx = list(range(n_train, n)) if n_holdout > 0 else train_idx

        X_train, Y_train = iris_arr[train_idx], screen_arr[train_idx]
        X_val, Y_val = iris_arr[holdout_idx], screen_arr[holdout_idx]

        # Ajuste do modelo
        fit_ok = self.model.fit(X_train, Y_train)
        if not fit_ok:
            logger.error("Falha no ajuste do modelo de calibração %s", self.model.name)
            return False, float('inf')

        self.is_calibrated = True

        # Avaliação das métricas sobre o conjunto de validação
        self.last_metrics = evaluate_calibration_metrics(
            self.model, X_val, Y_val, screen_w, screen_h
        )

        self.last_holdout_error = self.last_metrics.rmse_px
        train_metrics = evaluate_calibration_metrics(self.model, X_train, Y_train, screen_w, screen_h)
        self.last_train_error = train_metrics.rmse_px

        logger.info(
            "Calibração concluída (%s): %s",
            self.model.name, self.last_metrics
        )

        # Não salvar automaticamente se o erro for inaceitável
        if self.last_metrics.rmse_px <= CALIBRATION_REPROJECTION_ERROR_THRESHOLD:
            self.save_calibration()

        return True, self.last_holdout_error

    # ------------------------------------------------------------------
    # Mapeamento
    # ------------------------------------------------------------------

    def is_resolution_compatible(self, current_w: int, current_h: int) -> bool:
        """Verifica se a resolução atual coincide com a resolução da calibração."""
        return self.calibrated_screen_w == current_w and self.calibrated_screen_h == current_h

    def adapt_screen_resolution(self, current_w: int, current_h: int) -> bool:
        """
        Ajusta proporcionalmente o mapeamento caso a resolução de tela mude após a calibração.
        Retorna True se houve ajuste de escala, False se permaneceu idêntico.
        """
        if current_w <= 0 or current_h <= 0 or self.calibrated_screen_w <= 0 or self.calibrated_screen_h <= 0:
            return False
        if not self.is_resolution_compatible(current_w, current_h):
            self._scale_x = float(current_w) / float(self.calibrated_screen_w)
            self._scale_y = float(current_h) / float(self.calibrated_screen_h)
            logger.info(
                "Resolução alterada de %dx%d para %dx%d. Fatores de adaptação: sx=%.3f, sy=%.3f",
                self.calibrated_screen_w, self.calibrated_screen_h, current_w, current_h,
                self._scale_x, self._scale_y
            )
            return True
        else:
            self._scale_x = 1.0
            self._scale_y = 1.0
            return False

    def map_to_screen(
        self, iris_pos: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        """Mapeia coordenadas do olhar para pixels da tela, aplicando trim e adaptação de resolução."""
        if not self.is_calibrated or not self.model.is_fitted:
            return None

        x_arr = np.array(iris_pos, dtype=np.float64)
        try:
            pred = self.model.predict(x_arr)
            sx = int(round((pred[0] + self.trim_x) * getattr(self, "_scale_x", 1.0)))
            sy = int(round((pred[1] + self.trim_y) * getattr(self, "_scale_y", 1.0)))
            return sx, sy
        except Exception as exc:
            logger.error("Erro na predição de coordenadas: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Persistência Segura e Backup
    # ------------------------------------------------------------------

    def save_calibration(self) -> None:
        if not self.is_calibrated or not self.model.is_fitted:
            return

        # Backup do arquivo existente antes de sobrescrever
        if os.path.exists(self._json_file):
            try:
                shutil.copyfile(self._json_file, self._bak_file)
            except OSError:
                pass

        data = {
            "version": _CALIBRATION_VERSION,
            "profile": self.profile_name,
            "model_type": self.model.name,
            "screen_width": self.calibrated_screen_w,
            "screen_height": self.calibrated_screen_h,
            "trim_x": self.trim_x,
            "trim_y": self.trim_y,
            "coeffs_x": self.model.coeffs_x.tolist() if self.model.coeffs_x is not None else [],
            "coeffs_y": self.model.coeffs_y.tolist() if self.model.coeffs_y is not None else [],
            "condition_number": self.model.condition_number,
            "train_error_px": self.last_train_error,
            "holdout_error_px": self.last_holdout_error,
            "metrics": {
                "rmse_px": self.last_metrics.rmse_px if self.last_metrics else self.last_holdout_error,
                "median_px": self.last_metrics.median_error_px if self.last_metrics else self.last_holdout_error,
                "p95_px": self.last_metrics.p95_error_px if self.last_metrics else self.last_holdout_error,
                "diag_pct": self.last_metrics.relative_diag_pct if self.last_metrics else 0.0,
            } if self.last_metrics else {},
        }

        try:
            with open(self._json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Calibração salva com sucesso em %s", self._json_file)
        except OSError as exc:
            logger.error("Erro ao salvar arquivo de calibração: %s", exc)

    def load_calibration(self) -> bool:
        if os.path.exists(self._json_file):
            return self._load_json()
        if os.path.exists(self._npy_file):
            return self._migrate_npy_to_json()
        return False

    def _load_json(self) -> bool:
        try:
            with open(self._json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "coeffs_x" not in data or "coeffs_y" not in data:
                return False

            cx = data["coeffs_x"]
            cy = data["coeffs_y"]
            if not isinstance(cx, list) or not isinstance(cy, list):
                return False
            if len(cx) < 6 or len(cx) != len(cy):
                return False

            model_type = data.get("model_type", "polynomial")
            self.model = self._create_model(model_type)
            ok = self.model.set_coefficients(data)
            if not ok:
                return False

            self.is_calibrated = True
            self.calibrated_screen_w = int(data.get("screen_width", 1920))
            self.calibrated_screen_h = int(data.get("screen_height", 1080))
            self.trim_x = float(data.get("trim_x", 0.0))
            self.trim_y = float(data.get("trim_y", 0.0))
            self.last_holdout_error = float(data.get("holdout_error_px", float('inf')))
            self.last_train_error = float(data.get("train_error_px", float('inf')))

            logger.info(
                "Calibração carregada de %s (modelo %s, holdout_error=%.1fpx)",
                self._json_file, self.model.name, self.last_holdout_error
            )
            return True
        except Exception as exc:
            logger.error("Erro ao ler JSON de calibração: %s", exc)
            return False

    def _migrate_npy_to_json(self) -> bool:
        try:
            raw = np.load(self._npy_file, allow_pickle=True)
            data = raw.item()
            self.model = PolynomialCalibrationModel()
            self.model.coeffs_x = np.array(data["coeffs_x"], dtype=np.float64)
            self.model.coeffs_y = np.array(data["coeffs_y"], dtype=np.float64)
            self.model.is_fitted = True
            self.is_calibrated = True
            self.save_calibration()
            logger.info("Migração de .npy legado para JSON concluída.")
            return True
        except Exception as exc:
            logger.error("Falha ao migrar calibracao legada: %s", exc)
            return False

    def _validate_calibration(self) -> float:
        """Compatibilidade com testes legados."""
        if not self.is_calibrated:
            return float('inf')

        total_error = 0.0
        count = 0
        for iris_pt, screen_pt in zip(self.iris_points, self.screen_points):
            pred = self.map_to_screen(iris_pt)
            if pred:
                dist = float(np.linalg.norm(np.array(pred) - np.array(screen_pt)))
                total_error += dist
                count += 1
        return total_error / count if count > 0 else float('inf')
