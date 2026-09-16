"""
test_calibration_m3.py — Testes das capacidades avançadas do CalibrationManager (Milestone 3).
"""
import pytest
import numpy as np
import tempfile
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from calibration import CalibrationManager, filter_outliers_mad


class TestCalibrationManagerM3:
    def test_filter_outliers_mad(self):
        # 10 pontos normais em torno de (0.5, 0.5) e 2 outliers extremos (piscada / artefato)
        normal_pts = [(0.5 + np.random.normal(0, 0.005), 0.5 + np.random.normal(0, 0.005)) for _ in range(10)]
        outliers = [(0.1, 0.1), (0.9, 0.9)]
        all_pts = normal_pts + outliers

        filtered = filter_outliers_mad(all_pts, threshold_mad=2.5)
        assert len(filtered) < len(all_pts)
        assert (0.1, 0.1) not in filtered
        assert (0.9, 0.9) not in filtered

    def test_deduplication_of_samples(self):
        manager = CalibrationManager(profile_name="test_dedup")
        # Inserir primeira amostra com frame_id=1
        ok1 = manager.add_point_sample(target_idx=0, iris_pos=(0.5, 0.5), screen_pos=(100, 100), frame_id=1)
        assert ok1 is True

        # Inserir com o mesmo frame_id=1 (deve ser rejeitado)
        ok2 = manager.add_point_sample(target_idx=0, iris_pos=(0.5, 0.5), screen_pos=(100, 100), frame_id=1)
        assert ok2 is False

        # Inserir com novo frame_id=2 (deve ser aceito)
        ok3 = manager.add_point_sample(target_idx=0, iris_pos=(0.51, 0.51), screen_pos=(100, 100), frame_id=2)
        assert ok3 is True

    def test_set_trim_offsets_prediction(self):
        manager = CalibrationManager(profile_name="test_trim")
        # Configurar modelo fictício
        manager.model.coeffs_x = np.array([0, 1000, 0, 0, 0, 0], dtype=np.float64)
        manager.model.coeffs_y = np.array([0, 0, 1000, 0, 0, 0], dtype=np.float64)
        manager.model.is_fitted = True
        manager.is_calibrated = True

        # Sem trim
        sx, sy = manager.map_to_screen((0.5, 0.5))
        assert sx == 500
        assert sy == 500

        # Aplicar trim (+15px horizontal, -20px vertical)
        manager.set_trim(15.0, -20.0)
        sx_trim, sy_trim = manager.map_to_screen((0.5, 0.5))
        assert sx_trim == 515
        assert sy_trim == 480

    def test_recalibrate_point(self):
        manager = CalibrationManager(profile_name="test_recalib")
        for i in range(8):
            manager.add_point((0.1 * i, 0.1 * i), (100 * i, 100 * i))

        ok, _ = manager.compute_calibration()
        assert ok is True

        # Recalibrar ponto de índice 2 com nova coordenada
        recalib_ok = manager.recalibrate_point(2, (0.22, 0.22), (220, 220))
        assert recalib_ok is True
        assert manager.iris_points[2] == (0.22, 0.22)
