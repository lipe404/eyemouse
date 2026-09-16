"""
test_calibration.py — Testes para CalibrationManager (Milestone 1).

Cobre: persistencia JSON, validacao de perfil, holdout validation,
migracao de .npy legado e carregamento de arquivo invalido.
"""
import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def make_calibration_points(n=20):
    """Gera n pontos sinteticos de calibracao distribuidos uniformemente."""
    iris_points = []
    screen_points = []
    for i in range(n):
        x = 0.1 + 0.8 * (i % 4) / 3
        y = 0.1 + 0.8 * (i // 4) / (n // 4 - 1 if n >= 8 else 1)
        iris_points.append((x, y))
        screen_points.append((int(x * 1920), int(y * 1080)))
    return iris_points, screen_points


# ---------------------------------------------------------------------------
# Validacao de perfil
# ---------------------------------------------------------------------------

class TestProfileValidation:
    def test_valid_profile_name_accepted(self):
        """Nomes validos devem ser aceitos sem erro."""
        from calibration import CalibrationManager
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager(profile_name="user_1")
        assert cm.profile_name == "user_1"

    def test_empty_profile_raises(self):
        """Nome vazio deve levantar ValueError."""
        from calibration import _validate_profile_name
        with pytest.raises(ValueError, match="vazio"):
            _validate_profile_name("")

    def test_path_traversal_raises(self):
        """Path traversal deve ser rejeitado."""
        from calibration import _validate_profile_name
        with pytest.raises(ValueError, match="invalido"):
            _validate_profile_name("../../etc/passwd")

    def test_slash_in_name_raises(self):
        from calibration import _validate_profile_name
        with pytest.raises(ValueError, match="invalido"):
            _validate_profile_name("user/name")

    def test_too_long_name_raises(self):
        from calibration import _validate_profile_name
        with pytest.raises(ValueError):
            _validate_profile_name("a" * 65)

    def test_hyphens_and_underscores_allowed(self):
        from calibration import _validate_profile_name
        assert _validate_profile_name("user-name_01") == "user-name_01"


# ---------------------------------------------------------------------------
# Compute calibration com holdout
# ---------------------------------------------------------------------------

class TestComputeCalibration:
    @pytest.fixture
    def cm(self):
        from calibration import CalibrationManager
        with patch('calibration.os.path.exists', return_value=False):
            manager = CalibrationManager("test")
        manager._json_file = "/tmp/test_calib.json"
        return manager

    def test_insufficient_points_returns_false(self, cm):
        """Com menos de 6 pontos, compute_calibration deve retornar False."""
        cm.add_point((0.5, 0.5), (960, 540))
        success, error = cm.compute_calibration()
        assert success is False

    def test_compute_succeeds_with_enough_points(self, cm):
        iris_pts, screen_pts = make_calibration_points(20)
        for ip, sp in zip(iris_pts, screen_pts):
            cm.add_point(ip, sp)
        with patch.object(cm, 'save_calibration'):
            success, error = cm.compute_calibration()
        assert success is True
        assert error >= 0

    def test_holdout_error_is_reported_not_train_error(self, cm):
        """O erro retornado deve ser o holdout_error, nao o train_error."""
        iris_pts, screen_pts = make_calibration_points(20)
        for ip, sp in zip(iris_pts, screen_pts):
            cm.add_point(ip, sp)
        with patch.object(cm, 'save_calibration'):
            success, reported_error = cm.compute_calibration()
        assert success
        # holdout_error deve bater com o retornado
        assert abs(cm.last_holdout_error - reported_error) < 0.001

    def test_coefficients_shape(self, cm):
        """Os coeficientes devem ter 6 elementos (modelo polinomial 2a ordem)."""
        iris_pts, screen_pts = make_calibration_points(20)
        for ip, sp in zip(iris_pts, screen_pts):
            cm.add_point(ip, sp)
        with patch.object(cm, 'save_calibration'):
            cm.compute_calibration()
        assert cm.coeffs_x.shape == (6,)
        assert cm.coeffs_y.shape == (6,)

    def test_map_to_screen_after_calibration(self, cm):
        """map_to_screen deve retornar coordenadas de tela apos calibracao."""
        iris_pts, screen_pts = make_calibration_points(20)
        for ip, sp in zip(iris_pts, screen_pts):
            cm.add_point(ip, sp)
        with patch.object(cm, 'save_calibration'):
            cm.compute_calibration()
        result = cm.map_to_screen((0.5, 0.5))
        assert result is not None
        x, y = result
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_map_to_screen_without_calibration_returns_none(self, cm):
        assert cm.map_to_screen((0.5, 0.5)) is None


# ---------------------------------------------------------------------------
# Persistencia JSON
# ---------------------------------------------------------------------------

class TestJsonPersistence:
    @pytest.fixture
    def cm_with_calibration(self):
        """CalibrationManager ja calibrado e pronto para salvar."""
        from calibration import CalibrationManager
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
        cm.coeffs_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        cm.coeffs_y = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        cm.is_calibrated = True
        cm.last_holdout_error = 25.0
        cm.last_train_error = 10.0
        cm._json_file = "/tmp/test_calib.json"
        return cm

    def test_save_creates_valid_json(self, cm_with_calibration, tmp_path):
        """save_calibration() deve criar um JSON valido com todos os campos."""
        cm = cm_with_calibration
        cm._json_file = str(tmp_path / "calib.json")
        cm.save_calibration()
        with open(cm._json_file) as f:
            data = json.load(f)
        assert data["version"] == 2
        assert data["profile"] == "test"
        assert len(data["coeffs_x"]) == 6
        assert len(data["coeffs_y"]) == 6
        assert data["holdout_error_px"] == 25.0

    def test_load_valid_json(self, cm_with_calibration, tmp_path):
        """load_calibration() deve restaurar coeficientes corretamente."""
        from calibration import CalibrationManager
        cm = cm_with_calibration
        path = str(tmp_path / "calib.json")
        cm._json_file = path
        cm.save_calibration()

        # Criar novo manager e carregar
        with patch('calibration.os.path.exists', side_effect=lambda p: p == path):
            cm2 = CalibrationManager("test")
            cm2._json_file = path
        result = cm2._load_json()
        assert result is True
        assert cm2.is_calibrated is True
        np.testing.assert_array_almost_equal(cm2.coeffs_x, cm.coeffs_x)
        np.testing.assert_array_almost_equal(cm2.coeffs_y, cm.coeffs_y)
        assert cm2.last_holdout_error == 25.0

    def test_load_invalid_json_returns_false(self, tmp_path):
        """JSON invalido nao deve crashar; deve retornar False."""
        from calibration import CalibrationManager
        path = str(tmp_path / "bad_calib.json")
        with open(path, "w") as f:
            f.write("{ this is not valid json }")
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
            cm._json_file = path
        result = cm._load_json()
        assert result is False
        assert cm.is_calibrated is False

    def test_load_json_with_wrong_shape_returns_false(self, tmp_path):
        """JSON com coeficientes de shape errado nao deve ser aceito."""
        from calibration import CalibrationManager
        path = str(tmp_path / "wrong_shape.json")
        data = {"version": 2, "profile": "test",
                "coeffs_x": [1.0, 2.0], "coeffs_y": [1.0, 2.0]}
        with open(path, "w") as f:
            json.dump(data, f)
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
            cm._json_file = path
        result = cm._load_json()
        assert result is False

    def test_load_json_missing_field_returns_false(self, tmp_path):
        """JSON sem campo obrigatorio retorna False."""
        from calibration import CalibrationManager
        path = str(tmp_path / "missing_field.json")
        data = {"version": 2, "profile": "test",
                "coeffs_x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}  # coeffs_y faltando
        with open(path, "w") as f:
            json.dump(data, f)
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
            cm._json_file = path
        result = cm._load_json()
        assert result is False

    def test_save_does_nothing_when_not_calibrated(self, tmp_path):
        """save_calibration nao cria arquivo se nao calibrado."""
        from calibration import CalibrationManager
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
        cm._json_file = str(tmp_path / "should_not_exist.json")
        cm.save_calibration()
        assert not os.path.exists(cm._json_file)


# ---------------------------------------------------------------------------
# Pontos duplicados e deduplicacao (P4 - coleta thread-safe)
# ---------------------------------------------------------------------------

class TestCalibrationPoints:
    def test_add_point_stores_correctly(self):
        from calibration import CalibrationManager
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
        cm.add_point((0.3, 0.4), (500, 400))
        assert len(cm.iris_points) == 1
        assert cm.iris_points[0] == (0.3, 0.4)

    def test_clear_points(self):
        from calibration import CalibrationManager
        with patch('calibration.os.path.exists', return_value=False):
            cm = CalibrationManager("test")
        cm.add_point((0.5, 0.5), (960, 540))
        cm.clear_points()
        assert len(cm.iris_points) == 0
        assert len(cm.screen_points) == 0
