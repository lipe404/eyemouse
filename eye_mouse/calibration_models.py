"""
calibration_models.py — Modelos de calibração intercambiáveis e métricas de validação.

Milestone 3:
  - BaseCalibrationModel: Interface abstrata para regressores.
  - LinearCalibrationModel: Regressão linear pura.
  - PolynomialCalibrationModel: Regressão polinomial de 2ª ordem clássica.
  - RidgeCalibrationModel: Regressão com regularização L2 (Tikhonov) para estabilidade
    numérica contra matrizes degeneradas ou colineares.
  - Verificação rigorosa do condicionamento da matriz (np.linalg.cond).
  - Cálculo de métricas completas: Média, Mediana, RMSE, P95, Erro Regional e % da Diagonal.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Limite para considerar uma matriz mal-condicionada
MAX_CONDITION_NUMBER = 1e8


@dataclass(frozen=True)
class CalibrationMetrics:
    """Métricas abrangentes de erro de calibração."""
    mean_error_px: float
    median_error_px: float
    rmse_px: float
    p95_error_px: float
    relative_diag_pct: float         # Erro médio expresso em % da diagonal da tela
    regional_errors: Dict[str, float] # Erro por quadrante: "Q1", "Q2", "Q3", "Q4", "center"
    num_samples: int
    condition_number: float
    is_valid: bool

    def __str__(self) -> str:
        return (
            f"RMSE: {self.rmse_px:.1f}px | Média: {self.mean_error_px:.1f}px | "
            f"Mediana: {self.median_error_px:.1f}px | P95: {self.p95_error_px:.1f}px | "
            f"Diagonal: {self.relative_diag_pct:.2f}% | Cond: {self.condition_number:.1e}"
        )


class BaseCalibrationModel(ABC):
    """Interface abstrata para modelos de mapeamento de olhar para tela."""

    def __init__(self, name: str):
        self.name = name
        self.coeffs_x: Optional[np.ndarray] = None
        self.coeffs_y: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.condition_number: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> bool:
        """
        Ajusta os parâmetros do modelo.
        X: matriz (N, D) de features
        Y: matriz (N, 2) de coordenadas de tela [x, y]
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prediz coordenadas na tela para um vetor de features x.
        x: vetor (D,) ou matriz (N, D)
        Retorna: array (2,) ou (N, 2)
        """
        pass

    def get_coefficients(self) -> Dict[str, List[float]]:
        """Retorna os coeficientes em formato serializável para JSON."""
        return {
            "coeffs_x": self.coeffs_x.tolist() if self.coeffs_x is not None else [],
            "coeffs_y": self.coeffs_y.tolist() if self.coeffs_y is not None else [],
            "model_name": self.name,
            "condition_number": self.condition_number,
        }

    def set_coefficients(self, data: Dict[str, Any]) -> bool:
        """Carrega coeficientes de um dicionário JSON."""
        try:
            self.coeffs_x = np.array(data["coeffs_x"], dtype=np.float64)
            self.coeffs_y = np.array(data["coeffs_y"], dtype=np.float64)
            self.condition_number = float(data.get("condition_number", 0.0))
            self.is_fitted = True
            return True
        except Exception as exc:
            logger.error("Falha ao carregar coeficientes no modelo %s: %s", self.name, exc)
            return False


class LinearCalibrationModel(BaseCalibrationModel):
    """Modelo de regressão linear simples com termo de bias."""

    def __init__(self):
        super().__init__("linear")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> bool:
        if len(X) < 3:
            return False

        # Adicionar coluna de 1s para bias se não presente
        if not np.allclose(X[:, 0], 1.0):
            A = np.column_stack([np.ones(len(X)), X])
        else:
            A = X

        try:
            self.condition_number = float(np.linalg.cond(A))
            self.coeffs_x, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
            self.coeffs_y, _, _, _ = np.linalg.lstsq(A, Y[:, 1], rcond=None)
            self.is_fitted = True
            return True
        except Exception as exc:
            logger.error("Erro no fit do LinearCalibrationModel: %s", exc)
            return False

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Modelo não ajustado.")

        if x.ndim == 1:
            features = x if np.isclose(x[0], 1.0) else np.insert(x, 0, 1.0)
            return np.array([float(np.dot(features, self.coeffs_x)), float(np.dot(features, self.coeffs_y))])
        else:
            A = x if np.allclose(x[:, 0], 1.0) else np.column_stack([np.ones(len(x)), x])
            return np.column_stack([A @ self.coeffs_x, A @ self.coeffs_y])


class PolynomialCalibrationModel(BaseCalibrationModel):
    """Modelo polinomial de 2ª ordem [1, x, y, xy, x^2, y^2]."""

    def __init__(self):
        super().__init__("polynomial")

    def _expand_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            # Se já for vetor expandido, retornar
            if len(X) == 6 and np.isclose(X[0], 1.0):
                return X
            x, y = X[0], X[1]
            return np.array([1.0, x, y, x * y, x**2, y**2], dtype=np.float64)
        else:
            if X.shape[1] == 6 and np.allclose(X[:, 0], 1.0):
                return X
            x, y = X[:, 0], X[:, 1]
            ones = np.ones(len(X))
            return np.column_stack([ones, x, y, x * y, x**2, y**2])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> bool:
        if len(X) < 6:
            return False

        A = self._expand_features(X)
        try:
            self.condition_number = float(np.linalg.cond(A))
            self.coeffs_x, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
            self.coeffs_y, _, _, _ = np.linalg.lstsq(A, Y[:, 1], rcond=None)
            self.is_fitted = True
            return True
        except Exception as exc:
            logger.error("Erro no fit do PolynomialCalibrationModel: %s", exc)
            return False

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Modelo não ajustado.")

        A = self._expand_features(x)
        if A.ndim == 1:
            return np.array([float(np.dot(A, self.coeffs_x)), float(np.dot(A, self.coeffs_y))])
        else:
            return np.column_stack([A @ self.coeffs_x, A @ self.coeffs_y])


class RidgeCalibrationModel(BaseCalibrationModel):
    """
    Modelo com regularização L2 (Ridge / Tikhonov).
    Resolve (A^T A + lambda * I)^(-1) A^T Y, garantindo estabilidade mesmo com
    pontos próximos, colineares ou matriz mal-condicionada.
    """

    def __init__(self, alpha_l2: float = 1e-3, use_polynomial: bool = True):
        super().__init__("ridge")
        self.alpha_l2 = alpha_l2
        self.use_polynomial = use_polynomial

    def _prepare_design_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.use_polynomial and X.shape[-1] == 2:
            x, y = X[:, 0], X[:, 1]
            return np.column_stack([np.ones(len(X)), x, y, x * y, x**2, y**2])
        elif not np.allclose(X[:, 0], 1.0):
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X: np.ndarray, Y: np.ndarray) -> bool:
        n = len(X)
        if n < 4:
            return False

        A = self._prepare_design_matrix(X)
        d = A.shape[1]

        try:
            self.condition_number = float(np.linalg.cond(A))
            # Regularização L2: Não penalizar o termo de intercepto (índice 0)
            reg = self.alpha_l2 * np.eye(d, dtype=np.float64)
            reg[0, 0] = 0.0

            AtA = A.T @ A + reg
            AtY_x = A.T @ Y[:, 0]
            AtY_y = A.T @ Y[:, 1]

            self.coeffs_x = np.linalg.solve(AtA, AtY_x)
            self.coeffs_y = np.linalg.solve(AtA, AtY_y)
            self.is_fitted = True
            return True
        except Exception as exc:
            logger.warning("Falha na solução Ridge exata (%s). Tentando lstsq...", exc)
            try:
                self.coeffs_x, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
                self.coeffs_y, _, _, _ = np.linalg.lstsq(A, Y[:, 1], rcond=None)
                self.is_fitted = True
                return True
            except Exception as exc2:
                logger.error("Falha total no fit do RidgeCalibrationModel: %s", exc2)
                return False

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Modelo não ajustado.")

        if x.ndim == 1:
            X_2d = x.reshape(1, -1)
            A = self._prepare_design_matrix(X_2d)
            return np.array([float(np.dot(A[0], self.coeffs_x)), float(np.dot(A[0], self.coeffs_y))])
        else:
            A = self._prepare_design_matrix(x)
            return np.column_stack([A @ self.coeffs_x, A @ self.coeffs_y])


def evaluate_calibration_metrics(
    model: BaseCalibrationModel,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> CalibrationMetrics:
    """
    Avalia a calibração com métricas científicas completas sobre conjunto de validação separado.
    """
    n = len(X_val)
    if n == 0 or not model.is_fitted:
        return CalibrationMetrics(
            mean_error_px=float('inf'),
            median_error_px=float('inf'),
            rmse_px=float('inf'),
            p95_error_px=float('inf'),
            relative_diag_pct=float('inf'),
            regional_errors={},
            num_samples=0,
            condition_number=model.condition_number,
            is_valid=False,
        )

    preds = model.predict(X_val)
    diffs = preds - Y_val
    euclidean_errors = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

    mean_err = float(np.mean(euclidean_errors))
    median_err = float(np.median(euclidean_errors))
    rmse = float(np.sqrt(np.mean(euclidean_errors**2)))
    p95_err = float(np.percentile(euclidean_errors, 95))

    diag_px = math.sqrt(screen_w**2 + screen_h**2)
    relative_diag_pct = float((mean_err / diag_px) * 100.0)

    # Erro por quadrantes da tela
    cx, cy = screen_w / 2.0, screen_h / 2.0
    regional: Dict[str, List[float]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": [], "center": []}

    for i in range(n):
        sx, sy = Y_val[i, 0], Y_val[i, 1]
        err = euclidean_errors[i]

        # Região central (zona de 40% central)
        if abs(sx - cx) < screen_w * 0.2 and abs(sy - cy) < screen_h * 0.2:
            regional["center"].append(err)

        if sx >= cx and sy < cy:
            regional["Q1"].append(err)
        elif sx < cx and sy < cy:
            regional["Q2"].append(err)
        elif sx < cx and sy >= cy:
            regional["Q3"].append(err)
        else:
            regional["Q4"].append(err)

    regional_means = {
        k: float(np.mean(v)) if len(v) > 0 else mean_err for k, v in regional.items()
    }

    is_valid = (
        not math.isnan(rmse)
        and not math.isinf(rmse)
        and model.condition_number < MAX_CONDITION_NUMBER
    )

    return CalibrationMetrics(
        mean_error_px=mean_err,
        median_error_px=median_err,
        rmse_px=rmse,
        p95_error_px=p95_err,
        relative_diag_pct=relative_diag_pct,
        regional_errors=regional_means,
        num_samples=n,
        condition_number=model.condition_number,
        is_valid=is_valid,
    )
