"""
test_training_target_ui.py — Testes unitários para TrainingSession e TrainingTargetUI (Milestone 6).

Testa:
  - Inicialização da sessão com diferentes resoluções e contagens de alvos.
  - Sequência balanceada com alvos grandes (60px), médios (40px) e pequenos (25px).
  - Cálculo de distância euclidiana e detecção de acerto/erro.
  - Registro de tempos de reação e avanço sequencial.
  - Tratamento de cliques falsos após conclusão ou fora da sessão.
  - Cálculo de sumário estatístico e recomendações personalizadas.
  - Prevenção de divisão por zero em sessões vazias.
  - Integração da UI Tkinter em ambiente mock/headless.
"""
from unittest.mock import MagicMock, patch
import pytest

from eye_mouse.ui.training_target_ui import TargetAttempt, TrainingSession, TrainingSummary, TrainingTargetUI


class TestTrainingSession:
    def test_session_initialization(self):
        session = TrainingSession(screen_w=1920, screen_h=1080, num_targets=8)
        assert session.screen_w == 1920
        assert session.screen_h == 1080
        assert len(session.targets) == 8
        assert session.current_index == 0
        assert not session.is_completed
        assert len(session.attempts) == 0

    def test_target_radii_distribution(self):
        session = TrainingSession(screen_w=1920, screen_h=1080, num_targets=8)
        radii = [t[2] for t in session.targets]
        # Deve conter raios grande (60), médio (40) e pequeno (25)
        assert 60 in radii
        assert 40 in radii
        assert 25 in radii

    def test_record_click_exact_center_hit(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=3)
        session.start(timestamp=10.0)
        cx, cy, r = session.current_target

        finished, attempt = session.record_click(cx, cy, timestamp=10.5)
        assert not finished
        assert attempt.is_hit is True
        assert attempt.distance_px == 0.0
        assert attempt.time_to_target_sec == 0.5
        assert attempt.target_index == 0
        assert session.current_index == 1

    def test_record_click_inside_radius_hit(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=3)
        session.start(timestamp=1.0)
        cx, cy, r = session.current_target

        # Clique dentro do raio
        finished, attempt = session.record_click(cx + r // 2, cy, timestamp=1.2)
        assert attempt.is_hit is True
        assert attempt.distance_px == float(r // 2)

    def test_record_click_outside_radius_miss(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=3)
        session.start(timestamp=1.0)
        cx, cy, r = session.current_target

        # Clique fora do raio
        finished, attempt = session.record_click(cx + r + 20, cy, timestamp=1.3)
        assert attempt.is_hit is False
        assert attempt.distance_px == float(r + 20)

    def test_full_session_completion(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=2)
        session.start(timestamp=0.0)

        # Alvo 1
        cx1, cy1, r1 = session.current_target
        fin1, att1 = session.record_click(cx1, cy1, timestamp=0.5)
        assert not fin1
        assert not session.is_completed

        # Alvo 2
        cx2, cy2, r2 = session.current_target
        fin2, att2 = session.record_click(cx2, cy2, timestamp=1.0)
        assert fin2
        assert session.is_completed

        summary = session.get_summary()
        assert summary.total_targets == 2
        assert summary.hits == 2
        assert summary.targets_hit == 2
        assert summary.misses == 0
        assert summary.hit_rate_pct == 100.0
        assert "Excelente controle" in summary.recommendation

    def test_record_click_after_completion_counts_false_clicks(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=1)
        session.start(timestamp=0.0)
        cx, cy, r = session.current_target
        session.record_click(cx, cy, timestamp=0.5)
        assert session.is_completed

        # Tentativa após encerramento
        fin, att = session.record_click(100, 100, timestamp=1.0)
        assert fin is True
        assert session.false_clicks == 1

    def test_summary_moderate_score(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=4)
        session.start(timestamp=0.0)
        # 3 acertos, 1 erro -> 75%
        for i in range(3):
            cx, cy, _ = session.current_target
            session.record_click(cx, cy, timestamp=float(i + 1))
        # 4º erro
        cx, cy, r = session.current_target
        session.record_click(cx + r + 50, cy, timestamp=5.0)

        summary = session.get_summary()
        assert summary.hit_rate_pct == 75.0
        assert "Bom controle" in summary.recommendation

    def test_summary_low_score(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=4)
        session.start(timestamp=0.0)
        # 1 acerto, 3 erros -> 25%
        cx, cy, _ = session.current_target
        session.record_click(cx, cy, timestamp=1.0)
        for i in range(3):
            cx, cy, r = session.current_target
            session.record_click(cx + r + 50, cy, timestamp=float(i + 2))

        summary = session.get_summary()
        assert summary.hit_rate_pct == 25.0
        assert "Taxa de acerto baixa" in summary.recommendation

    def test_summary_empty_session_safe(self):
        session = TrainingSession(screen_w=1000, screen_h=1000, num_targets=5)
        summary = session.get_summary()
        assert summary.total_targets == 0
        assert summary.hit_rate_pct == 0.0
        assert "Nenhuma tentativa" in summary.recommendation


class TestTrainingTargetUIHeadless:
    def test_init_without_root(self):
        ui = TrainingTargetUI(root=None)
        assert ui.window is None
        assert ui.session is None

    @patch("tkinter.Toplevel")
    @patch("tkinter.Canvas")
    @patch("tkinter.Frame")
    @patch("tkinter.Label")
    def test_init_with_mock_root(self, mock_lbl_cls, mock_frame_cls, mock_canvas_cls, mock_toplevel_cls):
        mock_root = MagicMock()
        mock_win = MagicMock()
        mock_win.winfo_screenwidth.return_value = 1280
        mock_win.winfo_screenheight.return_value = 720
        mock_win.winfo_exists.return_value = True
        mock_toplevel_cls.return_value = mock_win

        mock_canvas = MagicMock()
        mock_canvas_cls.return_value = mock_canvas

        on_complete = MagicMock()
        on_cancel = MagicMock()

        ui = TrainingTargetUI(
            root=mock_root,
            on_complete=on_complete,
            on_cancel=on_cancel,
            num_targets=4,
        )

        assert ui.window is not None
        assert ui.session is not None
        assert len(ui.session.targets) == 4

        # Simula clique do usuário no centro do alvo atual
        cx, cy, r = ui.session.current_target
        event = MagicMock()
        event.x = cx
        event.y = cy

        ui._on_canvas_click(event)
        assert ui.session.current_index == 1

        # Fechar janela antes de concluir deve acionar cancel callback
        ui.close()
        on_cancel.assert_called_once()
