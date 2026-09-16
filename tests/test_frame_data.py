"""
test_frame_data.py — Testes para as estruturas FramePacket e TrackingResult.
"""
import time
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from frame_data import FramePacket, TrackingResult


class TestFramePacket:
    def test_creation_and_fields(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        now = time.perf_counter()
        packet = FramePacket(
            frame_id=42,
            capture_timestamp=now,
            image=img,
            camera_metadata={"backend": "DSHOW", "fps": 30},
        )
        assert packet.frame_id == 42
        assert packet.capture_timestamp == now
        assert packet.image.shape == (100, 100, 3)
        assert packet.camera_metadata["backend"] == "DSHOW"

    def test_age_and_expiration(self):
        old_time = time.perf_counter() - 0.200  # 200 ms atrás
        packet = FramePacket(
            frame_id=1,
            capture_timestamp=old_time,
            image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        assert packet.age_sec >= 0.190
        assert packet.is_expired(max_age_sec=0.150) is True
        assert packet.is_expired(max_age_sec=0.500) is False

    def test_immutability(self):
        packet = FramePacket(
            frame_id=1,
            capture_timestamp=time.perf_counter(),
            image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        with pytest.raises(AttributeError):
            packet.frame_id = 99


class TestTrackingResult:
    def test_valid_result(self):
        t0 = time.perf_counter()
        t1 = t0 + 0.015  # 15 ms de inferência
        res = TrackingResult(
            frame_id=10,
            capture_timestamp=t0,
            processed_timestamp=t1,
            landmarks=["dummy_landmark"],
            features={"avg_iris": np.array([0.5, 0.5])},
            tracking_valid=True,
            tracking_quality=0.95,
            is_stabilized=True,
        )
        assert res.frame_id == 10
        assert res.tracking_valid is True
        assert res.is_stabilized is True
        assert np.isclose(res.latency_sec, 0.015, atol=0.005)
        assert res.is_expired(0.500) is False

    def test_invalid_convenience_constructor(self):
        t0 = time.perf_counter()
        res = TrackingResult.invalid(
            frame_id=5,
            capture_ts=t0,
            reason="Face occluded",
        )
        assert res.frame_id == 5
        assert res.tracking_valid is False
        assert res.landmarks is None
        assert res.features == {}
        assert res.error_message == "Face occluded"
        assert res.is_stabilized is False
