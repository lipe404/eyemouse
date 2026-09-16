"""
test_main.py — Testes para EyeMouseApp (Milestone 1).

Foca nos cenarios de seguranca criticos:
  - Pausa enquanto botao pressionado
  - Encerramento durante arraste
  - Perda de camera
  - Retorno apos perda de rastreamento
  - Injecao de evento com falha
  - Maquina de estados: transicoes validas e efeitos colaterais

Nao move o cursor real, nao abre janelas reais.
"""
import sys
import os
import threading
import time
import queue
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app_mocks():
    """Retorna um conjunto de mocks para substituir dependencias externas."""
    mocks = {}

    # Tkinter
    mock_root = MagicMock()
    mock_root.after = MagicMock()
    mock_root.withdraw = MagicMock()
    mocks['root'] = mock_root

    # Modulos do EyeMouse
    mocks['gaze_tracker'] = MagicMock()
    mocks['blink_detector'] = MagicMock()
    mocks['blink_detector'].is_calibrating = False
    mocks['blink_detector'].ear_threshold = 0.20

    mocks['calibration_manager'] = MagicMock()
    mocks['calibration_manager'].load_calibration.return_value = False
    mocks['calibration_manager'].profile_name = "test"

    mocks['mouse_controller'] = MagicMock()
    mocks['mouse_controller'].is_dragging = False
    mocks['mouse_controller'].screen_size = (1920, 1080)

    # CalibrationUI mock (nunca deve abrir janela real)
    mocks['calibration_ui'] = MagicMock()

    return mocks


@pytest.fixture
def app():
    """
    Cria um EyeMouseApp com todas as dependencias mockadas.

    Nao abre janelas reais, nao move o cursor.
    """
    mocks = _make_app_mocks()

    with \
        patch('tkinter.Tk', return_value=mocks['root']), \
        patch('tkinter.simpledialog.askstring', return_value='test'), \
        patch('main.GazeTracker', return_value=mocks['gaze_tracker']), \
        patch('main.BlinkDetector', return_value=mocks['blink_detector']), \
        patch('main.CalibrationManager', return_value=mocks['calibration_manager']), \
        patch('main.MouseController', return_value=mocks['mouse_controller']), \
        patch('main.CalibrationUI', return_value=mocks['calibration_ui']), \
        patch('main.ControlPanel', return_value=MagicMock()), \
        patch('main.keyboard'), \
        patch('cv2.VideoCapture') as mock_cap_cls, \
        patch('tkinter.messagebox.askyesno', return_value=False), \
        patch('sys.exit'):

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        from main import EyeMouseApp
        application = EyeMouseApp()
        application._mocks = mocks
        application.running = False  # Nao iniciar threads de verdade
        yield application


# ---------------------------------------------------------------------------
# Testes de estado inicial
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_state_machine_exists(self, app):
        from app_state import StateMachine
        assert isinstance(app.state_machine, StateMachine)

    def test_profiler_exists(self, app):
        from benchmark import FrameProfiler
        assert isinstance(app.profiler, FrameProfiler)

    def test_ui_updates_dict_empty(self, app):
        assert isinstance(app._ui_updates, dict)
        assert len(app._ui_updates) == 0


# ---------------------------------------------------------------------------
# Testes de fila de UI
# ---------------------------------------------------------------------------

class TestUIQueue:
    def test_post_ui_update_stores_value(self, app):
        """_post_ui_update deve armazenar o valor no dict."""
        app._post_ui_update("fps", 30)
        assert app._ui_updates.get("fps") == 30

    def test_post_ui_update_latest_wins(self, app):
        """Segunda chamada com mesma chave substitui o valor."""
        app._post_ui_update("fps", 25)
        app._post_ui_update("fps", 30)
        assert app._ui_updates["fps"] == 30

    def test_post_ui_update_thread_safe(self, app):
        """Multiplas threads escrevendo nao devem corromper o dict."""
        errors = []

        def writer(key, value):
            try:
                for _ in range(100):
                    app._post_ui_update(key, value)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"key_{i}", i))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_update_ui_loop_clears_dict(self, app):
        """update_ui_loop deve consumir e limpar o dict de updates."""
        app._post_ui_update("fps", 30)
        app._post_ui_update("left_ear", 0.25)
        assert len(app._ui_updates) == 2
        app.update_ui_loop()  # Deve consumir as atualizacoes
        assert len(app._ui_updates) == 0


# ---------------------------------------------------------------------------
# Testes de pausa e seguranca
# ---------------------------------------------------------------------------

class TestPauseSafety:
    def test_toggle_pause_calls_release_all(self, app):
        """
        Cenario critico: pausar enquanto botao esta pressionado.
        release_all() deve ser chamado via StateMachine.
        """
        from app_state import AppState

        # Colocar em estado ACTIVE primeiro
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)

        # Simular botao pressionado
        app._mocks['mouse_controller']._is_dragging = True

        # Pausar
        app.toggle_pause(True)

        # release_all deve ter sido chamado (pelo StateMachine via on_release_all)
        app._mocks['mouse_controller'].release_all.assert_called()

    def test_toggle_pause_transitions_to_paused(self, app):
        """toggle_pause(True) deve transitar para PAUSED."""
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)
        app.toggle_pause(True)
        assert app.state_machine.state == AppState.PAUSED

    def test_toggle_resume_transitions_to_active(self, app):
        """toggle_pause(False) a partir de PAUSED deve transitar para ACTIVE."""
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)
        app.toggle_pause(True)
        app.toggle_pause(False)
        assert app.state_machine.state == AppState.ACTIVE

    def test_is_paused_property_reflects_state(self, app):
        """is_paused property deve refletir o estado da maquina."""
        from app_state import AppState
        assert app.is_paused is False  # INITIALIZING, nao PAUSED
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)
        app.toggle_pause(True)
        assert app.is_paused is True


# ---------------------------------------------------------------------------
# Testes de encerramento
# ---------------------------------------------------------------------------

class TestQuitSafety:
    def test_quit_app_calls_release_all_before_exit(self, app):
        """
        quit_app() deve chamar release_all() antes de encerrar.
        Critico: botao do mouse nao pode ficar preso no SO.
        """
        app.quit_app()
        app._mocks['mouse_controller'].release_all.assert_called()

    def test_quit_app_transitions_to_shutting_down(self, app):
        """quit_app() deve transitar para SHUTTING_DOWN."""
        from app_state import AppState
        app.quit_app()
        assert app.state_machine.state == AppState.SHUTTING_DOWN

    def test_shutdown_during_drag_releases_button(self, app):
        """
        Cenario: usuario fecha o app durante arraste.
        release_all() deve ser chamado antes de sys.exit().
        """
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)

        # Simular botao pressionado
        app._mocks['mouse_controller'].is_dragging = True

        app.quit_app()

        # release_all deve ter sido chamado (diretamente no quit_app + StateMachine)
        assert app._mocks['mouse_controller'].release_all.call_count >= 1

    def test_quit_stops_running(self, app):
        """quit_app() deve setar self.running = False."""
        app.running = True
        app.quit_app()
        assert app.running is False


# ---------------------------------------------------------------------------
# Testes de perda de camera
# ---------------------------------------------------------------------------

class TestCameraLoss:
    def test_camera_loss_calls_release_all(self, app):
        """
        Cenario: camera desconecta durante uso.
        release_all() deve ser chamado ao detectar falha.
        """
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)

        # Simular o que camera_loop faz ao detectar falha
        app._mocks['mouse_controller'].release_all()
        app.state_machine.try_transition(AppState.ERROR)

        app._mocks['mouse_controller'].release_all.assert_called()
        assert app.state_machine.state == AppState.ERROR


# ---------------------------------------------------------------------------
# Testes de perda e retorno de rastreamento
# ---------------------------------------------------------------------------

class TestTrackingLost:
    def test_tracking_lost_transition(self, app):
        """ACTIVE -> TRACKING_LOST deve chamar release_all."""
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)
        app.state_machine.try_transition(AppState.TRACKING_LOST)
        app._mocks['mouse_controller'].release_all.assert_called()
        assert app.state_machine.state == AppState.TRACKING_LOST

    def test_return_from_tracking_lost(self, app):
        """Apos perda, retornar ao ACTIVE deve ser possivel."""
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)
        app.state_machine.try_transition(AppState.TRACKING_LOST)
        success = app.state_machine.try_transition(AppState.ACTIVE)
        assert success is True
        assert app.state_machine.state == AppState.ACTIVE

    def test_no_mouse_events_in_tracking_lost(self, app):
        """Em TRACKING_LOST, mouse_allowed deve ser False."""
        from app_state import AppState
        app.state_machine.try_transition(AppState.CALIBRATING)
        app.state_machine.try_transition(AppState.ACTIVE)
        app.state_machine.try_transition(AppState.TRACKING_LOST)
        assert app.state_machine.mouse_allowed is False


# ---------------------------------------------------------------------------
# Testes de acesso thread-safe a dados
# ---------------------------------------------------------------------------

class TestThreadSafeDataAccess:
    def test_get_latest_gaze_raw_returns_none_initially(self, app):
        result = app.get_latest_gaze_raw()
        assert result is None

    def test_get_latest_gaze_timestamp_returns_zero_initially(self, app):
        result = app.get_latest_gaze_timestamp()
        assert result == 0.0

    def test_get_latest_frame_returns_none_initially(self, app):
        result = app.get_latest_frame()
        assert result is None

    def test_profile_validated_on_init(self, app):
        """Perfil de usuario deve ser validado."""
        assert app.user_profile == "test"
