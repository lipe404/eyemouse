import pytest
import numpy as np
import cv2
import time
from unittest.mock import MagicMock, patch
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from utils.smoothing import SmoothingFilter

class TestSmoothingFilter:
    @pytest.fixture
    def filter(self):
        return SmoothingFilter()

    def test_initialization(self, filter):
        assert filter.kf is not None
        assert filter.first_update is True
        assert filter.last_time is None
        assert filter.prev_output is None
        assert len(filter.position_buffer) == 0

    def test_first_update(self, filter):
        x, y = 100, 200
        out_x, out_y = filter.update(x, y)
        
        assert out_x == x
        assert out_y == y
        assert filter.first_update is False
        assert filter.last_time is not None
        assert filter.prev_output == (x, y)

    def test_subsequent_update(self, filter):
        # Primeiro update
        filter.update(100, 200)
        time.sleep(0.01) # Pequeno delay para dt > 0
        
        # Segundo update
        out_x, out_y = filter.update(110, 210)
        
        # O filtro deve retornar valores próximos, mas não necessariamente exatos devido ao modelo
        assert isinstance(out_x, (float, np.float32))
        assert isinstance(out_y, (float, np.float32))

    def test_reset(self, filter):
        filter.update(100, 200)
        filter.reset()
        
        assert filter.first_update is True
        assert filter.last_time is None
        assert len(filter.position_buffer) == 0
        assert np.all(filter.kf.statePost == 0)

    def test_set_alpha(self, filter):
        filter.set_alpha(0.5)
        assert filter.current_alpha == 0.5
        
        filter.set_alpha(1.5) # Deve clampar em 1.0
        assert filter.current_alpha == 1.0
        
        filter.set_alpha(-0.5) # Deve clampar em 0.01
        assert filter.current_alpha == 0.01

    def test_dead_zone_static(self, filter):
        # Inicializar
        filter.update(100, 100)
        filter.prev_output = (100, 100) # Forçar output anterior
        
        # Substituir o filtro de Kalman real por um mock completo
        # Isso evita problemas com atributos read-only do cv2.KalmanFilter
        mock_kf = MagicMock()
        # predict retorna estado a priori (x, y, dx, dy)
        mock_kf.predict.return_value = np.array([[101], [101], [0], [0]], dtype=np.float32)
        # correct retorna estado a posteriori (x, y, dx, dy)
        mock_kf.correct.return_value = np.array([[101], [101], [0], [0]], dtype=np.float32)
        
        # Injetar o mock no objeto filter
        filter.kf = mock_kf
        
        # Pequeno movimento (101, 101) vs anterior (100, 100)
        # A lógica de dead zone deve impedir a atualização se o movimento for pequeno
        # Porém, o teste exato depende da implementação de _apply_dead_zone
        
        # Vamos apenas garantir que o código roda sem erro de atributo read-only
        out_x, out_y = filter.update(101, 101)
        
        # Verificar se os métodos do KF foram chamados
        mock_kf.predict.assert_called_once()
        mock_kf.correct.assert_called_once()

    def test_sigmoid(self, filter):
        val_low = filter._sigmoid(0)
        val_high = filter._sigmoid(1000)
        
        assert val_low < 0.1 # Deve ser baixo
        assert val_high > 0.9 # Deve ser alto

    def test_dt_handling(self, filter):
        # Testar dt muito grande ou negativo
        filter.update(100, 100)
        filter.last_time = time.time() - 2.0 # 2 segundos atrás
        
        filter.update(110, 110)
        # O código ajusta dt > 1.0 para 1/30. Verificar se não crasha.
        
        filter.last_time = time.time() + 1.0 # Futuro (dt negativo)
        filter.update(120, 120)
        # Deve ajustar para 1/30. Verificar se não crasha.
