"""
test_diagnostics.py — Testes unitários para o módulo de diagnóstico e benchmark reproduzível.
"""

from __future__ import annotations

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eye_mouse")))

from diagnostics import (
    get_hardware_info,
    get_environment_info,
    probe_camera_status,
    get_baseline_comparison,
    get_calibration_summary,
    run_diagnostics,
    save_diagnostic_report,
)


class TestDiagnostics:

    def test_get_hardware_info(self):
        hw = get_hardware_info()
        assert "os" in hw
        assert "architecture" in hw
        assert "cpu_count_logical" in hw
        assert hw["cpu_count_logical"] is None or hw["cpu_count_logical"] > 0

    def test_get_environment_info(self):
        env = get_environment_info()
        assert "python_version" in env
        assert "is_64bit" in env
        assert "dependencies" in env
        assert isinstance(env["dependencies"], dict)
        assert "mediapipe" in env["dependencies"]
        assert "numpy" in env["dependencies"]

    def test_probe_camera_status(self):
        cam = probe_camera_status()
        assert "camera_index" in cam
        assert "camera_available" in cam
        assert isinstance(cam["camera_available"], bool)

    def test_get_baseline_comparison(self):
        comp = get_baseline_comparison()
        assert len(comp) >= 8
        for row in comp:
            assert "dimensao" in row
            assert "m1_baseline" in row
            assert "m7_rc" in row
            assert "melhoria" in row

    def test_get_calibration_summary(self):
        calib = get_calibration_summary("non_existent_profile_xyz")
        assert "is_calibrated" in calib
        assert calib["is_calibrated"] is False

    def test_run_diagnostics(self):
        rep = run_diagnostics("default")
        assert "timestamp" in rep
        assert "hardware" in rep
        assert "environment" in rep
        assert "camera" in rep
        assert "configuration" in rep
        assert "baseline_comparison" in rep

    def test_save_diagnostic_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr("diagnostics.get_user_data_dir", lambda: str(tmp_path))
        rep = run_diagnostics("default")
        json_p, md_p = save_diagnostic_report(rep)

        assert os.path.exists(json_p)
        assert os.path.exists(md_p)

        # Valida integridade do JSON salvo
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["hardware"]["os"] == rep["hardware"]["os"]

        # Valida conteúdo do Markdown
        with open(md_p, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "# Relatório de Diagnóstico" in md_text
        assert "Baseline (Milestone 1)" in md_text
