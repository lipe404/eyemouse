import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
import sys

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from calibration import CalibrationManager

class TestCalibrationManager:
    @pytest.fixture
    def manager(self):
        # Usar um nome de perfil de teste para evitar sobrescrever arquivos reais
        return CalibrationManager(profile_name="test_profile")

    def test_initialization(self, manager):
        assert manager.profile_name == "test_profile"
        assert "test_profile.npy" in manager.calibration_file
        assert len(manager.iris_points) == 0
        assert len(manager.screen_points) == 0
        assert manager.is_calibrated is False

    def test_add_point(self, manager):
        manager.add_point((0.5, 0.5), (100, 100))
        assert len(manager.iris_points) == 1
        assert len(manager.screen_points) == 1
        assert manager.iris_points[0] == (0.5, 0.5)
        assert manager.screen_points[0] == (100, 100)

    def test_clear_points(self, manager):
        manager.add_point((0.5, 0.5), (100, 100))
        manager.clear_points()
        assert len(manager.iris_points) == 0
        assert len(manager.screen_points) == 0

    def test_compute_calibration_insufficient_points(self, manager):
        # Menos de 6 pontos
        for i in range(5):
            manager.add_point((0.5, 0.5), (100, 100))
        
        success, error = manager.compute_calibration()
        assert success is False
        assert error == 0.0

    @patch('numpy.save')
    def test_compute_calibration_success(self, mock_save, manager):
        # Adicionar 6 pontos fictícios
        # Simular uma relação linear simples: Screen = Iris * 1000
        for i in range(6):
            val = i / 10.0
            manager.add_point((val, val), (val * 1000, val * 1000))
            
        success, error = manager.compute_calibration()
        
        assert success is True
        assert manager.is_calibrated is True
        assert manager.coeffs_x is not None
        assert manager.coeffs_y is not None
        # O erro deve ser muito baixo para uma relação linear perfeita
        assert error < 1.0 
        mock_save.assert_called_once()

    def test_map_to_screen_not_calibrated(self, manager):
        manager.is_calibrated = False
        assert manager.map_to_screen((0.5, 0.5)) is None

    def test_map_to_screen_calibrated(self, manager):
        # Setup manual de calibração para teste determinístico
        manager.is_calibrated = True
        # Coeffs para: Screen = Iris * 100
        # Polinômio: [1, x, y, xy, x^2, y^2]
        # Queremos apenas termo x (índice 1) e y (índice 2)
        manager.coeffs_x = np.array([0, 100, 0, 0, 0, 0])
        manager.coeffs_y = np.array([0, 0, 100, 0, 0, 0])
        
        # Testar (0.5, 0.5) -> (50, 50)
        result = manager.map_to_screen((0.5, 0.5))
        assert result == (50, 50)

    @patch('os.path.exists')
    @patch('numpy.load')
    def test_load_calibration_success(self, mock_load, mock_exists, manager):
        mock_exists.return_value = True
        
        # Mock do retorno do np.load
        mock_data = MagicMock()
        mock_data.item.return_value = {
            "coeffs_x": np.array([1, 2, 3]),
            "coeffs_y": np.array([4, 5, 6])
        }
        mock_load.return_value = mock_data
        
        success = manager.load_calibration()
        
        assert success is True
        assert manager.is_calibrated is True
        assert np.array_equal(manager.coeffs_x, np.array([1, 2, 3]))
        assert np.array_equal(manager.coeffs_y, np.array([4, 5, 6]))

    @patch('os.path.exists')
    def test_load_calibration_not_found(self, mock_exists, manager):
        mock_exists.return_value = False
        success = manager.load_calibration()
        assert success is False

    @patch('os.path.exists')
    @patch('numpy.load')
    def test_load_calibration_error(self, mock_load, mock_exists, manager):
        mock_exists.return_value = True
        mock_load.side_effect = Exception("Corrupted file")
        
        success = manager.load_calibration()
        assert success is False

    def test_validate_calibration(self, manager):
        # Se não calibrado
        manager.is_calibrated = False
        assert manager._validate_calibration() == float('inf')
        
        # Se calibrado
        manager.is_calibrated = True
        # Setup fake points and coeffs
        manager.iris_points = [(0,0)]
        manager.screen_points = [(10,10)]
        
        # Mock map_to_screen para retornar (12, 12) -> erro sqrt(2^2 + 2^2) = sqrt(8) approx 2.82
        with patch.object(manager, 'map_to_screen', return_value=(12, 12)):
            error = manager._validate_calibration()
            assert 2.8 < error < 2.9
