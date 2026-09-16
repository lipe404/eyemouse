"""
tests/test_one_euro_filter.py — Testes unitários para o One Euro Filter (Milestone 4).
"""

import math
import os
import sys
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from utils.smoothing import (
    OneEuroFilter,
    PassThroughFilter,
    KalmanSmoothingFilter,
    create_smoothing_filter,
    _LowPassFilter1D,
    _compute_alpha,
)


class TestOneEuroFilter:
    def test_init_default_params(self):
        f = OneEuroFilter()
        assert f.min_cutoff == 1.0
        assert f.beta == 0.007
        assert f.d_cutoff == 1.0
        assert f.enable_deadzone is False

    def test_first_update_returns_raw_coordinates_zero_lag(self):
        f = OneEuroFilter()
        x, y = f.update(500.0, 300.0, timestamp=1.0)
        assert x == 500.0
        assert y == 300.0

    def test_fixation_noise_attenuation(self):
        """Em repouso, a variância do sinal filtrado deve ser substancialmente menor que a do sinal ruidoso."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        rng = np.random.RandomState(42)
        true_val = 500.0
        noise = rng.normal(0, 5.0, 60)
        noisy_pts = true_val + noise

        filtered_pts = []
        t = 0.0
        for val in noisy_pts:
            fx, _ = f.update(val, val, timestamp=t)
            filtered_pts.append(fx)
            t += 1.0 / 30.0

        raw_std = float(np.std(noisy_pts[10:]))
        filt_std = float(np.std(filtered_pts[10:]))
        assert filt_std < raw_std * 0.55, f"Esperado < 55% do desvio bruto: {filt_std} vs {raw_std}"

    def test_high_speed_saccade_instant_responsiveness(self):
        """Em movimento rápido (sacada), a frequência de corte deve aumentar e o lag deve ser mínimo."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
        # Fixação inicial
        f.update(100.0, 100.0, timestamp=0.0)
        f.update(100.0, 100.0, timestamp=0.033)

        # Salto brusco (sacada de 900px)
        out_x, out_y = f.update(1000.0, 1000.0, timestamp=0.066)
        # O filtro deve acompanhar a maior parte do salto rapidamente (> 70% no primeiro frame de sacada)
        assert out_x > 700.0
        assert out_y > 700.0

    def test_dt_handling_zero_or_negative(self):
        """Se timestamps forem idênticos ou decrescentes, o filtro não deve crashar nem gerar NaN/Inf."""
        f = OneEuroFilter()
        f.update(100.0, 200.0, timestamp=1.0)
        # dt = 0
        out_x1, out_y1 = f.update(105.0, 205.0, timestamp=1.0)
        assert not math.isnan(out_x1) and not math.isinf(out_x1)
        # dt negativo
        out_x2, out_y2 = f.update(110.0, 210.0, timestamp=0.5)
        assert not math.isnan(out_x2) and not math.isinf(out_x2)

    def test_dt_large_gap_triggers_reset(self):
        """Se houver uma pausa muito longa (> 1s), o filtro deve resetar sem salto brusco extrapolado."""
        f = OneEuroFilter()
        f.update(100.0, 100.0, timestamp=0.0)
        f.update(100.0, 100.0, timestamp=0.033)

        # Pausa de 3 segundos
        out_x, out_y = f.update(800.0, 800.0, timestamp=3.033)
        # Após reset por hiato, o novo ponto deve ser aceito diretamente
        assert out_x == 800.0
        assert out_y == 800.0

    def test_reset(self):
        f = OneEuroFilter()
        f.update(100.0, 100.0, timestamp=0.0)
        f.reset()
        assert f._last_time is None
        assert not f._x_filter.is_initialized

    def test_set_parameters(self):
        f = OneEuroFilter()
        f.set_parameters(min_cutoff=2.5, beta=0.02, d_cutoff=1.5)
        assert f.min_cutoff == 2.5
        assert f.beta == 0.02
        assert f.d_cutoff == 1.5

    def test_set_alpha_mapping(self):
        f = OneEuroFilter()
        f.set_alpha(0.5)
        assert 2.0 <= f.min_cutoff <= 3.0
        f.set_alpha(0.01)
        assert f.min_cutoff <= 0.3
        f.set_alpha(1.0)
        assert f.min_cutoff >= 4.8

    def test_deadzone_hysteresis(self):
        """Verifica histerese: movimentos pequenos dentro da zona morta de fixação são retidos."""
        f = OneEuroFilter(
            min_cutoff=1.0,
            beta=0.007,
            enable_deadzone=True,
            deadzone_rest=3.0,
            deadzone_release=6.0,
        )
        f.update(500.0, 500.0, timestamp=0.0)
        f.update(500.0, 500.0, timestamp=0.033)

        # Micro-tremor de 1.5px
        out_x, out_y = f.update(501.5, 501.0, timestamp=0.066)
        assert abs(out_x - 500.0) < 1.0

        # Grande movimento de 50px rompe a histerese
        out_x2, _ = f.update(550.0, 500.0, timestamp=0.100)
        assert out_x2 > 515.0


class TestLowPassFilter1D:
    def test_filter_first_value(self):
        lpf = _LowPassFilter1D()
        val = lpf.filter(42.0, alpha=0.5)
        assert val == 42.0
        assert lpf.last_filtered == 42.0

    def test_filter_subsequent_value(self):
        lpf = _LowPassFilter1D()
        lpf.filter(0.0, alpha=1.0)
        val = lpf.filter(10.0, alpha=0.3)
        assert abs(val - 3.0) < 1e-5

    def test_compute_alpha(self):
        a = _compute_alpha(1.0, 1.0 / 30.0)
        assert 0.0 < a < 1.0
        assert _compute_alpha(0.0, 0.033) == 1.0
        assert _compute_alpha(1.0, 0.0) == 1.0


class TestFilterFactory:
    def test_create_one_euro(self):
        f = create_smoothing_filter("ONE_EURO", min_cutoff=2.0)
        assert isinstance(f, OneEuroFilter)
        assert f.min_cutoff == 2.0

    def test_create_kalman(self):
        f = create_smoothing_filter("KALMAN")
        assert isinstance(f, KalmanSmoothingFilter)

    def test_create_passthrough(self):
        f = create_smoothing_filter("NONE")
        assert isinstance(f, PassThroughFilter)
        out_x, out_y = f.update(123.4, 567.8)
        assert out_x == 123.4
        assert out_y == 567.8

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Tipo de filtro desconhecido"):
            create_smoothing_filter("INVALID_FILTER")
