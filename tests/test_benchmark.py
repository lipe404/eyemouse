"""
tests/test_benchmark.py
=======================
Unit tests for FrameProfiler, StageStats, PipelineReport and helpers
in eye_mouse/benchmark.py.

No external mocking is needed — benchmark.py is pure Python.
Where real elapsed time is needed we use time.sleep(0.01) inside a
measure() context so the test stays deterministic enough without
requiring a real camera or OS.
"""

import sys
import os
import time

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from benchmark import (  # noqa: E402
    FrameProfiler,
    StageStats,
    PipelineReport,
    _percentile,
    _fps_from_timestamps,
)
from collections import deque


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def profiler() -> FrameProfiler:
    """Fresh FrameProfiler for each test."""
    return FrameProfiler()


# ---------------------------------------------------------------------------
# Tests: measure() context manager
# ---------------------------------------------------------------------------

class TestMeasureContext:
    """Tests for the profiler.measure() context manager."""

    def test_measure_context_records_sample(self, profiler):
        """Using profiler.measure('test_stage') as a context records exactly one sample."""
        with profiler.measure('test_stage'):
            time.sleep(0.01)

        stats = profiler.stats('test_stage')
        assert stats.count == 1, "Exactly one sample must be recorded after one measure() block"

    def test_measure_records_positive_elapsed(self, profiler):
        """The recorded elapsed time must be > 0 ms after a real sleep."""
        with profiler.measure('test_stage'):
            time.sleep(0.01)

        stats = profiler.stats('test_stage')
        assert stats.mean_ms > 0.0, "Elapsed time must be positive"


# ---------------------------------------------------------------------------
# Tests: stats() accuracy
# ---------------------------------------------------------------------------

class TestStageStats:
    """Tests for FrameProfiler.stats() statistical correctness."""

    def test_stats_returns_correct_values(self, profiler):
        """Manually record 10 known ns values; verify mean, min, max."""
        stage = 'test_known'
        # Record 10 values: 1 ms .. 10 ms in nanoseconds
        ns_values = [i * 1_000_000 for i in range(1, 11)]  # 1e6 .. 10e6 ns
        for v in ns_values:
            profiler._record(stage, v)

        stats = profiler.stats(stage)

        assert stats.count == 10
        assert abs(stats.mean_ms - 5.5) < 0.001, f"Expected mean=5.5ms, got {stats.mean_ms}"
        assert abs(stats.min_ms - 1.0) < 0.001,  f"Expected min=1.0ms, got {stats.min_ms}"
        assert abs(stats.max_ms - 10.0) < 0.001, f"Expected max=10.0ms, got {stats.max_ms}"

    def test_stats_insufficient(self, profiler):
        """stats() on an empty stage returns StageStats with count==0."""
        stats = profiler.stats('nonexistent_stage')
        assert stats.count == 0
        assert stats.mean_ms == 0
        assert stats.name == 'nonexistent_stage'

    def test_stage_stats_str_with_data(self, profiler):
        """str(StageStats) contains the stage name and numeric values when count >= 2."""
        stage = 'pipeline_step'
        for v in [1_000_000, 2_000_000]:  # 1ms, 2ms
            profiler._record(stage, v)

        stats = profiler.stats(stage)
        result = str(stats)

        assert stage in result, "Stage name must appear in __str__ output"
        assert 'mean=' in result, "__str__ must include mean"
        assert 'ms' in result, "__str__ must include 'ms' unit"

    def test_stage_stats_str_single_sample(self, profiler):
        """str(StageStats) with count==1 uses the short format (no percentiles)."""
        profiler._record('single', 5_000_000)
        stats = profiler.stats('single')
        result = str(stats)
        assert 'n=1' in result
        assert 'mean=' in result


# ---------------------------------------------------------------------------
# Tests: _percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    """Tests for the internal _percentile() helper function."""

    def test_percentile_single_element(self):
        """_percentile([5.0], 95) == 5.0 (only one data point)."""
        assert _percentile([5.0], 95) == 5.0

    def test_percentile_two_elements(self):
        """_percentile([0.0, 10.0], 50) == 5.0 (linear interpolation at midpoint)."""
        result = _percentile([0.0, 10.0], 50)
        assert abs(result - 5.0) < 1e-9, f"Expected 5.0, got {result}"

    def test_percentile_empty_list(self):
        """_percentile([], p) returns 0.0 for any p."""
        assert _percentile([], 50) == 0.0
        assert _percentile([], 99) == 0.0

    def test_percentile_p100(self):
        """_percentile at p=100 returns the last (max) element."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 100) == 5.0

    def test_percentile_p0(self):
        """_percentile at p=0 returns the first (min) element."""
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 0) == 1.0


# ---------------------------------------------------------------------------
# Tests: capture_fps / processing_fps
# ---------------------------------------------------------------------------

class TestFpsCalculation:
    """Tests for the FPS calculation properties."""

    def test_capture_fps_with_no_data(self, profiler):
        """capture_fps == 0.0 with no timestamps recorded."""
        assert profiler.capture_fps == 0.0

    def test_capture_fps_with_data(self, profiler):
        """Record 30 timestamps 33 ms apart; capture_fps should be ~30 fps (28-32)."""
        interval_ns = 33_333_333  # ~33.33 ms
        base = time.perf_counter_ns()
        for i in range(30):
            profiler._capture_times.append(base + i * interval_ns)

        fps = profiler.capture_fps
        assert 28.0 <= fps <= 32.0, f"Expected ~30 fps, got {fps:.2f}"


# ---------------------------------------------------------------------------
# Tests: dropped_frames
# ---------------------------------------------------------------------------

class TestDroppedFrames:
    """Tests for the record_dropped_frame() counter."""

    def test_dropped_frames_counter(self, profiler):
        """record_dropped_frame() increments the counter; 5 calls -> dropped_frames == 5."""
        for _ in range(5):
            profiler.record_dropped_frame()
        assert profiler.dropped_frames == 5


# ---------------------------------------------------------------------------
# Tests: face detection rate
# ---------------------------------------------------------------------------

class TestFaceDetectionRate:
    """Tests for the face_detection_rate property."""

    def test_face_detection_rate(self, profiler):
        """10 frames (5 with face detected) -> face_detection_rate == 0.5."""
        for i in range(10):
            profiler.record_process_fps(face_detected=(i % 2 == 0))
        assert abs(profiler.face_detection_rate - 0.5) < 0.01

    def test_face_detection_rate_no_frames(self, profiler):
        """face_detection_rate == 0.0 when no frames have been processed."""
        assert profiler.face_detection_rate == 0.0


# ---------------------------------------------------------------------------
# Tests: report()
# ---------------------------------------------------------------------------

class TestReport:
    """Tests for FrameProfiler.report()."""

    def test_report_contains_all_stages(self, profiler):
        """report() has an entry for every stage in FrameProfiler.ALL_STAGES."""
        report = profiler.report()
        for stage in FrameProfiler.ALL_STAGES:
            assert stage in report.stages, f"Stage '{stage}' missing from report"

    def test_pipeline_report_str(self, profiler):
        """str(PipelineReport) contains FPS info and the measurement note."""
        report = profiler.report()
        result = str(report)

        assert 'FPS' in result or 'fps' in result.lower(), "__str__ must mention FPS"
        assert 'latencia' in result.lower() or 'latency' in result.lower() or 'AVISO' in result, (
            "__str__ must include the measurement note"
        )

    def test_report_capture_fps_zero_with_no_data(self, profiler):
        """report() capture_fps is 0.0 when no capture timestamps were recorded."""
        report = profiler.report()
        assert report.capture_fps == 0.0


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------

class TestReset:
    """Tests for FrameProfiler.reset()."""

    def test_reset_clears_all(self, profiler):
        """After adding samples and calling reset(), all counts must be 0."""
        # Add data to every canonical stage
        for stage in FrameProfiler.ALL_STAGES:
            profiler._record(stage, 1_000_000)
        profiler.record_dropped_frame()
        profiler.record_process_fps(face_detected=True)

        profiler.reset()

        for stage in FrameProfiler.ALL_STAGES:
            assert profiler.stats(stage).count == 0, (
                f"count for stage '{stage}' must be 0 after reset()"
            )
        assert profiler.dropped_frames == 0
        assert profiler.face_detection_rate == 0.0


# ---------------------------------------------------------------------------
# Tests: measure all canonical stages
# ---------------------------------------------------------------------------

class TestCanonicalStages:
    """Tests exercising all ALL_STAGES via measure() context."""

    def test_measure_all_canonical_stages(self, profiler):
        """All stages in ALL_STAGES must have count >= 1 after one measure() each."""
        for stage in FrameProfiler.ALL_STAGES:
            with profiler.measure(stage):
                pass  # zero-sleep is fine — perf_counter_ns will still move

        for stage in FrameProfiler.ALL_STAGES:
            assert profiler.stats(stage).count >= 1, (
                f"Stage '{stage}' must have at least 1 sample after measure()"
            )
