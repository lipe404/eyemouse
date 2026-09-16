"""
test_stale_frame_prevention.py — Validação de que frames antigos ou não estabilizados
NUNCA conseguem gerar movimentos ou cliques no cursor do mouse.
"""
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from frame_data import FramePacket, TrackingResult
from tracking_validator import TrackingValidator
from app_state import AppState


class TestStaleFrameMovementPrevention:
    """
    Testes de segurança rigorosos garantindo que frames fora da janela de validade
    não alcancem os atuadores de movimento ou clique.
    """

    @pytest.fixture
    def mock_components(self):
        mouse = MagicMock()
        mouse.move = MagicMock()
        mouse.left_click = MagicMock()
        mouse.right_click = MagicMock()
        mouse.double_click = MagicMock()
        mouse.start_drag = MagicMock()

        validator = TrackingValidator(
            max_observation_age_sec=0.150,
            stabilization_frames=5,
            stabilization_time_sec=0.050,
        )

        calib = MagicMock()
        calib.map_to_screen.return_value = (500, 500)

        blink = MagicMock()
        blink.is_calibrating = False
        # Simula detecção de piscada esquerda (que geraria clique)
        blink.process.return_value = (True, False, False, False, False, (0.1, 0.3))

        return mouse, validator, calib, blink

    def test_stale_frame_cannot_move_or_click(self, mock_components):
        mouse, validator, calib, blink = mock_components

        # Frame capturado há 300 ms (claramente acima do limite de 150 ms)
        stale_capture_time = time.perf_counter() - 0.300
        raw_result = TrackingResult(
            frame_id=1,
            capture_timestamp=stale_capture_time,
            processed_timestamp=time.perf_counter(),
            landmarks=[MagicMock(x=0.5, y=0.5) for _ in range(500)],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=1.0,
            is_stabilized=True,
        )

        # Passar pelo validador
        val = validator.validate(raw_result)

        # Regra de despacho de ação do main.py
        can_act = val.tracking_valid and val.is_stabilized and not val.is_expired(0.150)

        if can_act:
            mouse.move(500, 500)
            mouse.left_click()

        # O mouse JAMAIS deve ter sido movido ou clicado!
        assert can_act is False
        mouse.move.assert_not_called()
        mouse.left_click.assert_not_called()

    def test_unstabilized_frame_cannot_move_or_click(self, mock_components):
        mouse, validator, calib, blink = mock_components

        # Frame fresco (idade < 10ms), mas primeiro após recuperação (não estabilizado)
        fresh_time = time.perf_counter()
        raw_result = TrackingResult(
            frame_id=1,
            capture_timestamp=fresh_time,
            processed_timestamp=fresh_time + 0.005,
            landmarks=[MagicMock(x=0.5, y=0.5) for _ in range(500)],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=1.0,
        )

        val = validator.validate(raw_result)
        assert val.tracking_valid is True
        assert val.is_stabilized is False  # Primeiro frame: ainda em estabilização

        can_act = val.tracking_valid and val.is_stabilized and not val.is_expired(0.150)

        if can_act:
            mouse.move(500, 500)
            mouse.left_click()

        assert can_act is False
        mouse.move.assert_not_called()
        mouse.left_click.assert_not_called()

    def test_fresh_stabilized_frame_can_act(self, mock_components):
        mouse, validator, calib, blink = mock_components

        # Alimentar frames suficientes para completar estabilização
        time.sleep(0.060)
        for i in range(1, 6):
            res = TrackingResult(
                frame_id=i,
                capture_timestamp=time.perf_counter(),
                processed_timestamp=time.perf_counter(),
                landmarks=[MagicMock(x=0.5, y=0.5) for _ in range(500)],
                features={"avg_iris": np.array([0.5, 0.5])},
                tracking_valid=True,
                tracking_quality=1.0,
            )
            val = validator.validate(res)

        assert val.is_stabilized is True
        assert val.tracking_valid is True
        assert val.is_expired(0.150) is False

        can_act = val.tracking_valid and val.is_stabilized and not val.is_expired(0.150)
        assert can_act is True

        if can_act:
            mouse.move(500, 500)
            mouse.left_click()

        mouse.move.assert_called_once_with(500, 500)
        mouse.left_click.assert_called_once()
