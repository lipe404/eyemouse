"""
test_calibration_models.py — Testes dos regressores e métricas de calibração.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from calibration_models import (
    LinearCalibrationModel,
    PolynomialCalibrationModel,
    RidgeCalibrationModel,
    evaluate_calibration_metrics,
)


class TestCalibrationModels:
    @pytest.fixture
    def synthetic_data(self):
        # 16 alvos em grade uniforme
        xs = np.linspace(0.2, 0.8, 4)
        ys = np.linspace(0.2, 0.8, 4)
        X = np.array([[x, y] for y in ys for x in xs])
        # Y mapeia para tela 1920x1080 com relação linear + quadrática suave
        Y = np.column_stack([
            X[:, 0] * 1920 + 20 * X[:, 0]**2,
            X[:, 1] * 1080 + 15 * X[:, 1]**2,
        ])
        return X, Y

    def test_linear_model_fit_and_predict(self, synthetic_data):
        X, Y = synthetic_data
        model = LinearCalibrationModel()
        assert model.fit(X, Y) is True
        assert model.is_fitted is True

        pred = model.predict(X[0])
        assert pred.shape == (2,)
        assert np.linalg.norm(pred - Y[0]) < 100.0  # Aproximação razoável

    def test_polynomial_model_fit_and_predict(self, synthetic_data):
        X, Y = synthetic_data
        model = PolynomialCalibrationModel()
        assert model.fit(X, Y) is True

        preds = model.predict(X)
        assert preds.shape == Y.shape
        # O modelo polinomial de 2ª ordem deve reproduzir o sinal quase perfeitamente
        assert np.mean(np.linalg.norm(preds - Y, axis=1)) < 5.0

    def test_ridge_regularization_stability(self):
        """Testa estabilidade da regressão Ridge mesmo em matriz quase-singular."""
        # Pontos quase perfeitamente colineares
        X = np.array([[0.5, 0.5], [0.5001, 0.5001], [0.5002, 0.5002], [0.5003, 0.5003]])
        Y = np.array([[100, 100], [105, 105], [110, 110], [115, 115]])

        model = RidgeCalibrationModel(alpha_l2=0.01)
        assert model.fit(X, Y) is True
        pred = model.predict(np.array([0.5001, 0.5001]))
        assert not np.isnan(pred[0])
        assert not np.isinf(pred[0])

    def test_evaluate_calibration_metrics(self, synthetic_data):
        X, Y = synthetic_data
        model = PolynomialCalibrationModel()
        model.fit(X[:12], Y[:12])  # Treina em 12 pontos

        metrics = evaluate_calibration_metrics(model, X[12:], Y[12:], screen_w=1920, screen_h=1080)
        assert metrics.num_samples == 4
        assert metrics.rmse_px > 0.0
        assert metrics.p95_error_px >= metrics.median_error_px
        assert metrics.relative_diag_pct > 0.0
        assert "center" in metrics.regional_errors
        assert metrics.is_valid is True
