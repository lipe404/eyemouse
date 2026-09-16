"""
test_tracking_validator.py — Testes do validador temporal e estabilizador de rastreamento.
"""
import time
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from frame_data import TrackingResult
from tracking_validator import TrackingValidator


class TestTrackingValidator:
    def test_stale_observation_rejected(self):
        """Frames mais antigos que max_observation_age_sec devem ser marcados como inválidos."""
        validator = TrackingValidator(max_observation_age_sec=0.100)

        # Frame capturado 250ms atrás
        old_ts = time.perf_counter() - 0.250
        stale_result = TrackingResult(
            frame_id=1,
            capture_timestamp=old_ts,
            processed_timestamp=time.perf_counter(),
            landmarks=["dummy"],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=1.0,
            is_stabilized=True,
        )

        validated = validator.validate(stale_result)
        assert validated.tracking_valid is False
        assert "expirado" in (validated.error_message or "").lower()

    def test_non_increasing_frame_id_rejected(self):
        """Frames repetidos ou com ID decrescente não devem ser processados."""
        validator = TrackingValidator()
        now = time.perf_counter()

        res1 = TrackingResult(
            frame_id=10,
            capture_timestamp=now,
            processed_timestamp=now,
            landmarks=["dummy"],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=1.0,
        )
        val1 = validator.validate(res1)
        assert val1.frame_id == 10

        # Tentativa de reutilizar o mesmo frame_id 10
        res2 = TrackingResult(
            frame_id=10,
            capture_timestamp=now + 0.01,
            processed_timestamp=now + 0.01,
            landmarks=["dummy"],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=1.0,
        )
        val2 = validator.validate(res2)
        assert val2.tracking_valid is False
        assert "repetido" in (val2.error_message or "").lower()

    def test_stabilization_period_after_recovery(self):
        """Ao recuperar o rosto, os primeiros frames não devem ser estabilizados até atingir o limite."""
        validator = TrackingValidator(stabilization_frames=5, stabilization_time_sec=0.050)

        now = time.perf_counter()
        # Primeiros 4 frames devem ter is_stabilized == False
        for i in range(1, 5):
            res = TrackingResult(
                frame_id=i,
                capture_timestamp=now + (i * 0.010),
                processed_timestamp=now + (i * 0.010),
                landmarks=["dummy"],
                features={"avg_iris": np.array([0.5, 0.5])},
                tracking_valid=True,
                tracking_quality=1.0,
            )
            val = validator.validate(res)
            assert val.tracking_valid is True
            assert val.is_stabilized is False, f"Frame {i} não deveria estar estabilizado ainda"

        # Simular passagem do tempo de estabilização (50ms)
        time.sleep(0.060)

        # 5º frame com tempo decorrido deve estar estabilizado
        res5 = TrackingResult(
            frame_id=5,
            capture_timestamp=time.perf_counter(),
            processed_timestamp=time.perf_counter(),
            landmarks=["dummy"],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=1.0,
        )
        val5 = validator.validate(res5)
        assert val5.tracking_valid is True
        assert val5.is_stabilized is True

    def test_loss_of_face_resets_stabilization(self):
        """Perder o rosto reseta a contagem e exige nova estabilização."""
        validator = TrackingValidator(stabilization_frames=3, stabilization_time_sec=0.010)

        now = time.perf_counter()
        # Estabilizar com 3 frames
        for i in range(1, 4):
            res = TrackingResult(
                frame_id=i,
                capture_timestamp=time.perf_counter(),
                processed_timestamp=time.perf_counter(),
                landmarks=["dummy"],
                features={},
                tracking_valid=True,
            )
            val = validator.validate(res)
            time.sleep(0.012)

        assert val.is_stabilized is True

        # Perda do rosto (frame inválido)
        no_face = TrackingResult.invalid(frame_id=4, capture_ts=time.perf_counter(), reason="No face")
        val_lost = validator.validate(no_face)
        assert val_lost.tracking_valid is False
        assert val_lost.is_stabilized is False

        # Retorno do rosto imediato -> NÃO deve estar estabilizado no primeiro frame
        recovered = TrackingResult(
            frame_id=5,
            capture_timestamp=time.perf_counter(),
            processed_timestamp=time.perf_counter(),
            landmarks=["dummy"],
            features={},
            tracking_valid=True,
        )
        val_recovered = validator.validate(recovered)
        assert val_recovered.tracking_valid is True
        assert val_recovered.is_stabilized is False  # Precisa re-estabilizar!

    def test_reset(self):
        validator = TrackingValidator()
        validator._consecutive_valid_frames = 10
        validator._last_validated_frame_id = 99
        validator.reset()
        assert validator._consecutive_valid_frames == 0
        assert validator._last_validated_frame_id == -1
