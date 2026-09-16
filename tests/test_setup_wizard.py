"""
test_setup_wizard.py — Testes unitários para SetupWizardState e SetupWizardUI (Milestone 6).

Testa:
  - Transições sequenciais de etapas do assistente (9 passos).
  - Bloqueio estrito da etapa de calibração se inválida ou erro > 80px.
  - Liberação de avanço apenas com calibração precisa (< 80px).
  - Análise fotométrica de luminância (escuro, ideal, superexposto).
  - Comportamento de navegação anterior/posterior.
  - Inicialização headless e disparo de callbacks (cancelamento e ativação final).
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from eye_mouse.ui.setup_wizard import LightingQuality, SetupWizardState, SetupWizardUI, WizardStep


class TestSetupWizardState:
    def test_initial_state(self):
        state = SetupWizardState()
        assert state.current_step == WizardStep.WELCOME
        assert state.selected_camera_index == 0
        assert not state.calibration_valid
        assert state.holdout_error == 999.0
        assert state.selected_profile == "HYBRID"

    def test_step_progression(self):
        state = SetupWizardState()
        assert state.next_step() == WizardStep.CAMERA
        assert state.next_step() == WizardStep.POSITIONING
        assert state.next_step() == WizardStep.LIGHTING
        assert state.next_step() == WizardStep.CALIBRATION

    def test_prev_step_regression(self):
        state = SetupWizardState()
        state.next_step()  # CAMERA
        state.next_step()  # POSITIONING
        assert state.prev_step() == WizardStep.CAMERA
        assert state.prev_step() == WizardStep.WELCOME
        # Não regride além de WELCOME
        assert state.prev_step() == WizardStep.WELCOME

    def test_calibration_gating_blocks_uncalibrated(self):
        state = SetupWizardState()
        state.current_step = WizardStep.CALIBRATION
        state.calibration_valid = False

        can_go, msg = state.can_advance()
        assert can_go is False
        assert "obrigatório concluir" in msg

        # Tentar avançar não altera o passo
        assert state.next_step() == WizardStep.CALIBRATION

    def test_calibration_gating_blocks_high_error(self):
        state = SetupWizardState()
        state.current_step = WizardStep.CALIBRATION
        state.calibration_valid = True
        state.holdout_error = 95.0  # > 80.0px limite

        can_go, msg = state.can_advance()
        assert can_go is False
        assert state.next_step() == WizardStep.CALIBRATION

    def test_calibration_gating_allows_good_calibration(self):
        state = SetupWizardState()
        state.current_step = WizardStep.CALIBRATION
        state.calibration_valid = True
        state.holdout_error = 38.5  # < 80.0px excelente

        can_go, msg = state.can_advance()
        assert can_go is True
        assert state.next_step() == WizardStep.GESTURES


class TestLightingPhotometry:
    def test_analyze_lighting_none_or_empty(self):
        luma, qual = SetupWizardState.analyze_lighting(None)
        assert luma == 0.0
        assert qual == LightingQuality.TOO_DARK

        luma_empty, qual_empty = SetupWizardState.analyze_lighting(np.array([]))
        assert luma_empty == 0.0
        assert qual_empty == LightingQuality.TOO_DARK

    def test_analyze_lighting_too_dark(self):
        # Frame escuro com valor médio 25
        frame = np.full((100, 100, 3), 25, dtype=np.uint8)
        luma, qual = SetupWizardState.analyze_lighting(frame)
        assert luma < 40.0
        assert qual == LightingQuality.TOO_DARK

    def test_analyze_lighting_ideal(self):
        # Frame equilibrado com valor médio 120
        frame = np.full((100, 100, 3), 120, dtype=np.uint8)
        luma, qual = SetupWizardState.analyze_lighting(frame)
        assert 40.0 <= luma <= 200.0
        assert qual == LightingQuality.IDEAL

    def test_analyze_lighting_too_bright(self):
        # Frame superexposto com valor médio 230
        frame = np.full((100, 100, 3), 230, dtype=np.uint8)
        luma, qual = SetupWizardState.analyze_lighting(frame)
        assert luma > 200.0
        assert qual == LightingQuality.TOO_BRIGHT


class TestSetupWizardUIHeadless:
    def test_init_without_root(self):
        ui = SetupWizardUI(root=None)
        assert ui.window is None
        assert ui.state.current_step == WizardStep.WELCOME

    @patch("tkinter.Toplevel")
    @patch("tkinter.Frame")
    @patch("tkinter.Label")
    @patch("tkinter.ttk.Frame")
    @patch("tkinter.ttk.Button")
    @patch("tkinter.ttk.Label")
    def test_init_and_cancel_callback(self, *mocks):
        mock_root = MagicMock()
        mock_win = MagicMock()
        mock_win.winfo_screenwidth.return_value = 1920
        mock_win.winfo_screenheight.return_value = 1080
        mock_win.winfo_exists.return_value = True
        mocks[5].return_value = mock_win  # Toplevel mock

        on_cancel = MagicMock()
        on_finish = MagicMock()

        ui = SetupWizardUI(
            root=mock_root,
            on_cancel=on_cancel,
            on_finish=on_finish,
        )
        assert ui.window is not None

        # Simula acionamento do botão cancelar
        ui._on_cancel()
        on_cancel.assert_called_once()

    @patch("tkinter.Toplevel")
    @patch("tkinter.Frame")
    @patch("tkinter.Label")
    @patch("tkinter.ttk.Frame")
    @patch("tkinter.ttk.Button")
    @patch("tkinter.ttk.Label")
    def test_finish_activation_at_confirm_step(self, *mocks):
        mock_root = MagicMock()
        mock_win = MagicMock()
        mock_win.winfo_screenwidth.return_value = 1920
        mock_win.winfo_screenheight.return_value = 1080
        mock_win.winfo_exists.return_value = True
        mocks[5].return_value = mock_win

        on_finish = MagicMock()

        ui = SetupWizardUI(
            root=mock_root,
            on_finish=on_finish,
        )

        # Coloca o wizard no passo final de confirmação
        ui.state.current_step = WizardStep.CONFIRM
        ui.state.calibration_valid = True
        ui.state.holdout_error = 28.0
        ui.state.selected_profile = "DWELL"

        ui._on_next()

        on_finish.assert_called_once()
        settings = on_finish.call_args[0][0]
        assert settings["interaction"]["profile"] == "DWELL"
        assert settings["calibration"]["holdout_error"] == 28.0
        assert settings["calibration"]["is_calibrated"] is True
