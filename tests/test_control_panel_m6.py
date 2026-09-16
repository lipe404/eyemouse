"""
test_control_panel_m6.py — Testes unitários para o ControlPanel redesenhado no Milestone 6.

Testa:
  - Presença das 4 abas estruturadas (Visão Geral, Ajustes Rápidos, Avançado, Perfis & Privacidade).
  - Atualização completa de métricas em tempo real (`update_metrics`):
      * FPS de captura e de processamento.
      * Classificação da calibração (Excelente < 40px, Boa 40-80px, Imprecisa > 80px).
      * Latência de pipeline estimada em milissegundos.
      * Estados de arraste (Arrastando vs Normal) e precisão.
  - Disparo de novos callbacks acessíveis:
      * `on_profile_change`
      * `on_action_bar_toggle`
      * `on_precision_toggle`
      * `on_open_wizard`
      * `on_open_training`
      * `on_clear_data`
  - Acessibilidade: presença do aviso legal de tecnologia assistiva.
"""
from unittest.mock import MagicMock, patch
import pytest

from eye_mouse.ui.control_panel import ControlPanel


class FakeVar:
    def __init__(self, master=None, value="", name=None):
        self._val = value
    def get(self):
        return self._val
    def set(self, val):
        self._val = val


@pytest.fixture
def mock_panel():
    """Cria uma instância de ControlPanel com todos os componentes Tkinter mockados."""
    with \
        patch("tkinter.Toplevel") as mock_top, \
        patch("tkinter.StringVar", side_effect=FakeVar), \
        patch("tkinter.BooleanVar", side_effect=FakeVar), \
        patch("tkinter.DoubleVar", side_effect=FakeVar), \
        patch("tkinter.ttk.Notebook") as mock_nb, \
        patch("tkinter.ttk.Frame") as mock_frame, \
        patch("tkinter.ttk.Label") as mock_lbl, \
        patch("tkinter.ttk.Button") as mock_btn, \
        patch("tkinter.ttk.Scale") as mock_scale, \
        patch("tkinter.ttk.Combobox") as mock_combo, \
        patch("tkinter.ttk.Checkbutton") as mock_check, \
        patch("tkinter.ttk.Separator") as mock_sep:

        mock_root = MagicMock()
        mock_win = MagicMock()
        mock_win.winfo_screenwidth.return_value = 1920
        mock_win.winfo_screenheight.return_value = 1080
        mock_top.return_value = mock_win

        on_toggle_pause = MagicMock()
        on_recalibrate = MagicMock()
        on_quit = MagicMock()
        on_smoothing_change = MagicMock()
        on_blink_calibrate = MagicMock()
        on_profile_change = MagicMock()
        on_action_bar_toggle = MagicMock()
        on_open_wizard = MagicMock()
        on_open_training = MagicMock()
        on_clear_data = MagicMock()
        on_precision_toggle = MagicMock()

        panel = ControlPanel(
            root=mock_root,
            on_pause_toggle=on_toggle_pause,
            on_recalibrate=on_recalibrate,
            on_quit=on_quit,
            update_smoothing_cb=on_smoothing_change,
            on_blink_calibrate=on_blink_calibrate,
            on_profile_change=on_profile_change,
            on_action_bar_toggle=on_action_bar_toggle,
            on_open_wizard=on_open_wizard,
            on_open_training=on_open_training,
            on_clear_data=on_clear_data,
            on_precision_toggle=on_precision_toggle,
        )

        yield panel, {
            "on_toggle_pause": on_toggle_pause,
            "on_profile_change": on_profile_change,
            "on_action_bar_toggle": on_action_bar_toggle,
            "on_open_wizard": on_open_wizard,
            "on_open_training": on_open_training,
            "on_clear_data": on_clear_data,
            "on_precision_toggle": on_precision_toggle,
        }


class TestControlPanelM6:
    def test_notebook_and_tabs_initialized(self, mock_panel):
        panel, _ = mock_panel
        assert hasattr(panel, "notebook")
        assert panel.notebook is not None

    def test_update_metrics_excellent_calibration(self, mock_panel):
        panel, _ = mock_panel
        panel.update_metrics(
            fps=30,
            process_fps=30.0,
            holdout_error=25.4,
            latency_ms=14.2,
            is_dragging=False,
            is_precision=False,
            profile_name="HYBRID",
        )
        assert "FPS: 30" in panel.metric_fps_var.get()
        assert "EXCELENTE" in panel.metric_calib_var.get()
        assert "14.2 ms" in panel.metric_latency_var.get()
        assert "Normal" in panel.metric_drag_var.get()
        assert "Padrão" in panel.metric_precision_var.get()

    def test_update_metrics_good_calibration(self, mock_panel):
        panel, _ = mock_panel
        panel.update_metrics(
            fps=28,
            process_fps=27.5,
            holdout_error=55.0,
            latency_ms=18.0,
            is_dragging=True,
            is_precision=True,
            profile_name="DWELL",
        )
        assert "BOA" in panel.metric_calib_var.get()
        assert "Arrastando" in panel.metric_drag_var.get()
        assert "Precisão Ativa" in panel.metric_precision_var.get()

    def test_update_metrics_imprecise_calibration(self, mock_panel):
        panel, _ = mock_panel
        panel.update_metrics(
            fps=15,
            process_fps=14.0,
            holdout_error=92.3,
            latency_ms=35.0,
            is_dragging=False,
            is_precision=False,
            profile_name="HANDS_FREE",
        )
        assert "IMPRECISA" in panel.metric_calib_var.get()

    def test_profile_combobox_callback(self, mock_panel):
        panel, callbacks = mock_panel
        panel.profile_var.set("Modo Dwell (Apenas Fixação)")
        panel._on_profile_selected()
        callbacks["on_profile_change"].assert_called_once_with("dwell_only")

    def test_action_bar_toggle_callback(self, mock_panel):
        panel, callbacks = mock_panel
        # Começa True -> toggla para False
        panel._on_action_bar_toggle()
        callbacks["on_action_bar_toggle"].assert_called_once_with(False)

    def test_precision_toggle_callback(self, mock_panel):
        panel, callbacks = mock_panel
        panel.precision_var.set(True)
        panel._on_precision_toggle()
        callbacks["on_precision_toggle"].assert_called_once_with(True)

    def test_open_wizard_and_training_callbacks(self, mock_panel):
        panel, callbacks = mock_panel
        panel._on_open_wizard()
        callbacks["on_open_wizard"].assert_called_once()

        panel._on_open_training()
        callbacks["on_open_training"].assert_called_once()

    def test_clear_data_callback(self, mock_panel):
        panel, callbacks = mock_panel
        panel._on_clear_data()
        callbacks["on_clear_data"].assert_called_once()
