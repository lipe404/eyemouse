"""
main.py — Ponto de entrada e orquestrador do EyeMouse.

Otimizações do Milestone 2:
  - Integração com CameraCapture (buffer mínimo latest-frame, seleção de backends).
  - Integração com TrackingValidator (supressão de frames expirados e período de estabilização pós-perda).
  - Rastreamento estruturado com FramePacket e TrackingResult.
  - Separação estrita da última observação válida do último frame de preview visual.
  - Frequência de preview desacoplada (PREVIEW_FPS) e preview opcional (SHOW_PREVIEW).
  - Garantia formal de que frames atrasados ou não estabilizados não movem cursor nem geram cliques.
"""
from __future__ import annotations

import cv2
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import logging
import sys
import queue
import re
from typing import Any, Dict, Optional

try:
    import keyboard
except ImportError:
    class _KeyboardStub:
        def add_hotkey(self, *a, **kw): pass
        def unhook_all(self): pass
    keyboard = _KeyboardStub()  # type: ignore[assignment]

from gaze_tracker import GazeTracker
from blink_detector import BlinkDetector
from calibration import CalibrationManager
from mouse_controller import MouseController
from ui.calibration_ui import CalibrationUI
from ui.control_panel import ControlPanel
from app_state import AppState, StateMachine, InvalidTransition
from benchmark import FrameProfiler
from frame_data import FramePacket, TrackingResult
from tracking_validator import TrackingValidator
from camera_capture import CameraCapture
from gesture_engine import GestureEngine, MouseAction
from dwell_clicker import DwellClicker
from interaction_profiles import ProfileManager, InteractionProfileType
from scroll_controller import ScrollController
from ui.action_bar import ActionBar
from settings_manager import SettingsManager
from ui.setup_wizard import SetupWizardUI
from ui.training_target_ui import TrainingTargetUI, TrainingSummary

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS,
    CAMERA_BACKEND, CAMERA_FOURCC, CAMERA_BUFFER_SIZE,
    LOG_FILE, CALIBRATION_REPROJECTION_ERROR_THRESHOLD,
    TRACKING_LOST_TIMEOUT_SEC, BENCHMARK_MODE,
    MAX_OBSERVATION_AGE_SEC, TRACKING_STABILIZATION_FRAMES,
    TRACKING_STABILIZATION_TIME_SEC, SHOW_PREVIEW, PREVIEW_FPS, DEBUG_DRAW,
    INTERACTION_PROFILE, DWELL_TIME_SEC, DWELL_RADIUS_PIXELS,
    DWELL_REARM_DISTANCE_PIXELS, DWELL_REARM_TIMEOUT_SEC,
    CLICK_FREEZE_DURATION_SEC, BILATERAL_WINDOW_SEC,
    SCROLL_TOP_ZONE_RATIO, SCROLL_BOTTOM_ZONE_RATIO, SCROLL_TICK_INTERVAL_SEC,
    FT_STATE_MACHINE, FT_SAFETY_RELEASE, FT_LATEST_FRAME_QUEUE, FT_TRACKING_VALIDATOR,
    FT_GESTURE_ENGINE, FT_DWELL_CLICK, FT_ACTION_BAR, FT_SCROLL_MODE
)

try:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
except OSError:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

logger = logging.getLogger(__name__)


class EyeMouseApp:
    """
    Classe principal da aplicação EyeMouse.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        # --- Perfil de usuário ---
        raw_profile = simpledialog.askstring(
            "Perfil de Usuário",
            "Digite seu nome (ou deixe em branco para 'default'):",
            parent=self.root,
        )
        self.user_profile = raw_profile.strip() if raw_profile and raw_profile.strip() else "default"
        if not re.match(r'^[A-Za-z0-9_\-]{1,64}$', self.user_profile):
            logger.warning("Perfil '%s' inválido; usando 'default'.", self.user_profile)
            self.user_profile = "default"

        # --- Estado compartilhado (thread-safe) ---
        self.data_lock = threading.Lock()
        self.latest_gaze_raw: Optional[np.ndarray] = None
        self.latest_gaze_timestamp: float = 0.0
        self.latest_frame: Optional[np.ndarray] = None
        self.last_face_time: float = 0.0

        # Separação explícita da última observação válida vs último frame de preview
        self.latest_valid_observation: Optional[TrackingResult] = None
        self.latest_preview_frame: Optional[np.ndarray] = None
        self._last_preview_render_time: float = 0.0

        # Fila de frames (câmera -> processamento)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._legacy_frame_counter: int = 0

        # Fila de atualizações da UI (latest-wins)
        self._ui_updates: Dict[str, Any] = {}
        self._ui_lock = threading.Lock()

        # Controle de execução
        self.running: bool = False
        self.calibration_ui = None
        self.control_panel = None
        self.setup_wizard: Optional[SetupWizardUI] = None
        self.training_ui: Optional[TrainingTargetUI] = None

        # Benchmark e Validador Temporal
        self.profiler = FrameProfiler()
        self.validator = TrackingValidator(
            max_observation_age_sec=MAX_OBSERVATION_AGE_SEC,
            stabilization_frames=TRACKING_STABILIZATION_FRAMES,
            stabilization_time_sec=TRACKING_STABILIZATION_TIME_SEC,
        )

        # Hotkey global
        try:
            keyboard.add_hotkey("ctrl+shift+p", self._hotkey_toggle_pause)
        except Exception as exc:
            logger.error("Erro ao registrar hotkey: %s", exc)

        # Inicializar módulos de visão e controle
        try:
            self.gaze_tracker = GazeTracker()
            self.blink_detector = BlinkDetector()
            self.calibration_manager = CalibrationManager(profile_name=self.user_profile)
            self.mouse_controller = MouseController()
            logger.info("Módulos inicializados. Perfil: %s", self.user_profile)
        except Exception as exc:
            logger.error("Erro ao inicializar módulos: %s", exc)
            messagebox.showerror("Erro Fatal", f"Falha ao iniciar: {exc}")
            sys.exit(1)

        # Máquina de estados
        self.state_machine = StateMachine(
            initial=AppState.INITIALIZING,
            on_release_all=self.mouse_controller.release_all if FT_SAFETY_RELEASE else None,
        )

        # Módulos de Interação e Acessibilidade (Milestone 5 & 6)
        self.settings_manager = SettingsManager()
        self.user_settings = self.settings_manager.load_profile(self.user_profile)
        saved_profile = self.user_settings.get("interaction", {}).get("profile", INTERACTION_PROFILE)
        self.profile_manager = ProfileManager(initial_profile=saved_profile)
        self.gesture_engine = GestureEngine(
            click_freeze_sec=CLICK_FREEZE_DURATION_SEC,
            bilateral_window_sec=BILATERAL_WINDOW_SEC,
            enable_double_blink=self.profile_manager.active_config.enable_double_blink,
        )
        self.dwell_clicker = DwellClicker(
            dwell_time_sec=DWELL_TIME_SEC,
            dwell_radius_px=DWELL_RADIUS_PIXELS,
            rearm_distance_px=DWELL_REARM_DISTANCE_PIXELS,
            rearm_timeout_sec=DWELL_REARM_TIMEOUT_SEC,
        )
        self.scroll_controller = ScrollController(
            screen_h=self.mouse_controller.screen_h,
            top_zone_ratio=SCROLL_TOP_ZONE_RATIO,
            bottom_zone_ratio=SCROLL_BOTTOM_ZONE_RATIO,
            min_tick_interval_sec=SCROLL_TICK_INTERVAL_SEC,
        )
        self.action_bar: Optional[ActionBar] = None
        self._apply_profile_config()

        # Câmera e Captura (com suporte ao CameraCapture otimizado e fallback defensivo)
        self.camera_capture: Optional[CameraCapture] = None
        self.cap = None

        if FT_LATEST_FRAME_QUEUE:
            try:
                self.camera_capture = CameraCapture(
                    camera_index=CAMERA_INDEX,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=TARGET_FPS,
                    backend=CAMERA_BACKEND,
                    fourcc=CAMERA_FOURCC,
                    buffer_size=CAMERA_BUFFER_SIZE,
                )
                self.camera_capture.start()
                self.cap = self.camera_capture.cap
            except Exception as exc:
                logger.warning("Falha na inicialização do CameraCapture (%s). Tentando cv2.VideoCapture...", exc)

        if self.cap is None:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                logger.error("Câmera não encontrada (index %d).", CAMERA_INDEX)
                messagebox.showerror(
                    "Erro de Câmera",
                    f"Não foi possível acessar a câmera (index {CAMERA_INDEX}).",
                )
                sys.exit(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if BENCHMARK_MODE:
            logger.info("BENCHMARK_MODE ativo: pipeline roda mas cursor NÃO será movido.")

        # Iniciar fluxo de calibração ou controle
        if self.calibration_manager.load_calibration():
            use_calib = messagebox.askyesno(
                "Calibração Encontrada",
                f"Calibração encontrada para '{self.user_profile}'. Deseja usar?",
            )
            if use_calib:
                self._enter_active_after_calibration_load()
            else:
                self.start_calibration()
        else:
            self.start_calibration()

        # Iniciar threads
        self.running = True
        self.camera_thread = threading.Thread(
            target=self.camera_loop, daemon=True, name="camera"
        )
        self.camera_thread.start()

        self.processing_thread = threading.Thread(
            target=self.processing_loop, daemon=True, name="processing"
        )
        self.processing_thread.start()

        # Loop de atualização da UI
        self.root.after(100, self.update_ui_loop)

    # ------------------------------------------------------------------
    # Calibração e Perfis de Interação
    # ------------------------------------------------------------------

    def _apply_profile_config(self) -> None:
        """Sincroniza o perfil ativo com o GestureEngine e o DwellClicker."""
        cfg = self.profile_manager.active_config
        self.gesture_engine.set_gesture_enabled("left_blink", cfg.enable_left_blink)
        self.gesture_engine.set_gesture_enabled("right_blink", cfg.enable_right_blink)
        self.gesture_engine.set_gesture_enabled("double_blink", cfg.enable_double_blink)
        self.gesture_engine.set_gesture_enabled("hold_start", cfg.enable_hold_drag)
        self.gesture_engine.set_gesture_enabled("hold_end", cfg.enable_hold_drag)
        self.gesture_engine.set_gesture_enabled("dwell", cfg.enable_dwell_click)
        if not cfg.enable_dwell_click:
            self.dwell_clicker.reset()

    def _show_action_bar_if_enabled(self) -> None:
        """Inicializa ou exibe a barra de ações se permitida pelo perfil e flags."""
        if FT_ACTION_BAR and self.profile_manager.active_config.enable_action_bar:
            if self.action_bar is None:
                try:
                    self.action_bar = ActionBar(
                        root=self.root,
                        on_action_selected=self._on_action_bar_selected,
                        on_pause_toggle=self.toggle_pause,
                        on_scroll_toggle=self._on_scroll_toggle,
                        on_precision_toggle=self._on_precision_toggle,
                        on_drag_toggle=self._on_drag_toggle,
                    )
                except Exception as exc:
                    logger.warning("Falha ao inicializar ActionBar (%s).", exc)
            elif self.action_bar:
                self.action_bar.show()

    def _on_action_bar_selected(self, action: MouseAction) -> None:
        logger.info("Ação selecionada na barra: %s", action.name)

    def _on_scroll_toggle(self, is_scrolling: bool) -> None:
        if is_scrolling:
            self.scroll_controller.activate()
        else:
            self.scroll_controller.deactivate()

    def _on_precision_toggle(self, is_precision: bool) -> None:
        self.mouse_controller.set_precision_mode(is_precision)

    def _on_drag_toggle(self, is_dragging: bool) -> None:
        if is_dragging:
            self.mouse_controller.start_drag()
        else:
            self.mouse_controller.stop_drag()

    def _execute_mouse_action(self, action: MouseAction) -> None:
        """Executa a ação de mouse especificada de forma centralizada e segura."""
        if action == MouseAction.LEFT_CLICK:
            self.mouse_controller.left_click()
        elif action == MouseAction.RIGHT_CLICK:
            self.mouse_controller.right_click()
        elif action == MouseAction.DOUBLE_CLICK:
            self.mouse_controller.double_click()
        elif action == MouseAction.START_DRAG:
            self.mouse_controller.start_drag()
            if self.action_bar:
                self.action_bar.set_dragging(True)
        elif action == MouseAction.STOP_DRAG:
            self.mouse_controller.stop_drag()
            if self.action_bar:
                self.action_bar.set_dragging(False)
        elif action == MouseAction.TOGGLE_PRECISION:
            new_mode = self.mouse_controller.toggle_precision_mode()
            if self.action_bar:
                self.action_bar.set_precision(new_mode)
        elif action == MouseAction.SCROLL_UP:
            self.mouse_controller.scroll(120)
        elif action == MouseAction.SCROLL_DOWN:
            self.mouse_controller.scroll(-120)
        elif action == MouseAction.PAUSE:
            self.toggle_pause(not self.is_paused)

    def _enter_active_after_calibration_load(self):
        try:
            self.state_machine.transition(AppState.CALIBRATING)
            self.state_machine.transition(AppState.ACTIVE)
        except InvalidTransition:
            pass
        self.show_control_panel()
        self._show_action_bar_if_enabled()

    def start_calibration(self):
        logger.info("Iniciando calibração...")
        self.state_machine.try_transition(AppState.CALIBRATING)
        self.validator.reset()

        if self.control_panel:
            self.control_panel.window.withdraw()
        if self.action_bar:
            self.action_bar.hide()

        self.calibration_ui = CalibrationUI(
            self.root,
            self.calibration_manager,
            self.get_latest_gaze_raw,
            self.on_calibration_complete,
            self.get_latest_frame,
        )

    def start_blink_calibration(self):
        self.blink_detector.start_calibration(duration=10.0)
        messagebox.showinfo(
            "Calibração de Piscada",
            "Olhe para a tela e pisque normalmente por 10 segundos.\n"
            "O sistema ajustará a sensibilidade automaticamente.",
        )

    def on_calibration_complete(self, cancelled: bool = False):
        if cancelled:
            logger.info("Calibração cancelada.")
            self.state_machine.try_transition(AppState.ACTIVE)
            if self.control_panel:
                self.control_panel.window.deiconify()
            else:
                self.show_control_panel()
            self._show_action_bar_if_enabled()
            return

        success, error = self.calibration_manager.compute_calibration()

        if success:
            if error > CALIBRATION_REPROJECTION_ERROR_THRESHOLD:
                retry = messagebox.askyesno(
                    "Calibração Imprecisa",
                    f"Qualidade baixa (holdout error: {error:.1f}px).\n"
                    f"Recomendado: < {CALIBRATION_REPROJECTION_ERROR_THRESHOLD}px.\n\n"
                    "Deseja tentar novamente?",
                )
                if retry:
                    self.start_calibration()
                    return
            else:
                messagebox.showinfo(
                    "Sucesso",
                    f"Calibração concluída!\nErro médio (holdout): {error:.1f}px",
                )
        else:
            messagebox.showerror("Erro", "Falha ao computar calibração. Tente novamente.")
            self.start_calibration()
            return

        logger.info("Calibração concluída. holdout_error=%.1fpx", error)
        self.state_machine.try_transition(AppState.ACTIVE)

        if self.control_panel:
            self.control_panel.window.deiconify()
        else:
            self.show_control_panel()
        self._show_action_bar_if_enabled()

    def show_control_panel(self):
        if not self.control_panel:
            self.control_panel = ControlPanel(
                self.root,
                self.toggle_pause,
                self.start_calibration,
                self.quit_app,
                self.update_smoothing,
                self.start_blink_calibration,
                on_profile_change=self.on_profile_change_from_panel,
                on_action_bar_toggle=self.on_action_bar_toggle,
                on_open_wizard=self.open_setup_wizard,
                on_open_training=self.open_training_screen,
                on_clear_data=self.clear_user_data,
                on_precision_toggle=self._on_precision_toggle,
            )

    def on_profile_change_from_panel(self, profile_name: str) -> None:
        """Callback acionado quando o usuário altera o perfil de interação no ControlPanel."""
        try:
            self.profile_manager.set_profile(profile_name)
            self._apply_profile_config()
            if "interaction" not in self.user_settings:
                self.user_settings["interaction"] = {}
            self.user_settings["interaction"]["profile"] = profile_name
            self.settings_manager.save_profile(self.user_profile, self.user_settings)
            logger.info("Perfil de interação alterado para '%s' e salvo.", profile_name)
        except Exception as exc:
            logger.warning("Falha ao trocar perfil para '%s': %s", profile_name, exc)

    def on_action_bar_toggle(self, enabled: bool) -> None:
        """Callback para habilitar ou desabilitar a barra de ações."""
        if "accessibility" not in self.user_settings:
            self.user_settings["accessibility"] = {}
        self.user_settings["accessibility"]["enable_action_bar"] = enabled
        self.settings_manager.save_profile(self.user_profile, self.user_settings)

        if enabled:
            self._show_action_bar_if_enabled()
        else:
            if self.action_bar:
                self.action_bar.hide()

    def open_setup_wizard(self) -> None:
        """Abre o Assistente de Configuração Inicial (Setup Wizard)."""
        if self.control_panel and hasattr(self.control_panel, "window"):
            self.control_panel.window.withdraw()
        if self.action_bar:
            self.action_bar.hide()

        self.setup_wizard = SetupWizardUI(
            parent=self.root,
            on_finish=self.on_wizard_finish,
            on_cancel=self.on_wizard_cancel,
            get_frame_fn=self.get_latest_frame,
            on_start_calibration=self.start_calibration,
        )

    def on_wizard_finish(self, state_dict: Dict[str, Any]) -> None:
        """Callback chamado ao concluir com sucesso o assistente de configuração."""
        self.setup_wizard = None
        logger.info("Setup Wizard concluído com sucesso.")

        # Sincronizar perfil escolhido no wizard
        profile_type = state_dict.get("interaction_profile", "hybrid")
        self.on_profile_change_from_panel(profile_type)

        if self.control_panel and hasattr(self.control_panel, "window"):
            self.control_panel.window.deiconify()
        else:
            self.show_control_panel()
        self._show_action_bar_if_enabled()

    def on_wizard_cancel(self) -> None:
        """Callback chamado quando o assistente é cancelado."""
        self.setup_wizard = None
        logger.info("Setup Wizard cancelado.")
        if self.control_panel and hasattr(self.control_panel, "window"):
            self.control_panel.window.deiconify()
        else:
            self.show_control_panel()
        self._show_action_bar_if_enabled()

    def open_training_screen(self) -> None:
        """Abre a Tela de Treinamento com Alvos Interativos."""
        if self.control_panel and hasattr(self.control_panel, "window"):
            self.control_panel.window.withdraw()

        self.training_ui = TrainingTargetUI(
            parent=self.root,
            on_complete=self.on_training_complete,
            on_cancel=self.on_training_cancel,
        )

    def on_training_complete(self, summary: TrainingSummary) -> None:
        """Callback acionado ao término da sessão de treinamento."""
        self.training_ui = None
        logger.info(
            "Treinamento concluído. Acertos: %d/%d (%.1f%%)",
            summary.targets_hit, summary.total_targets, summary.hit_rate_pct
        )
        if self.control_panel and hasattr(self.control_panel, "window"):
            self.control_panel.window.deiconify()
        else:
            self.show_control_panel()

    def on_training_cancel(self) -> None:
        """Callback chamado se a tela de treinamento for cancelada."""
        self.training_ui = None
        if self.control_panel and hasattr(self.control_panel, "window"):
            self.control_panel.window.deiconify()
        else:
            self.show_control_panel()

    def clear_user_data(self) -> None:
        """Limpa todas as calibrações e perfis salvos do usuário (garantia de privacidade)."""
        confirm = messagebox.askyesno(
            "Confirmar Exclusão de Dados",
            "Tem certeza que deseja apagar todas as calibrações e perfis?\n\n"
            "Nenhuma gravação de vídeo é armazenada, mas todos os arquivos de configuração "
            "locais serão excluídos permanentemente.",
        )
        if confirm:
            self.settings_manager.clear_all_user_data()
            self.calibration_manager.clear_points()
            messagebox.showinfo(
                "Dados Excluídos",
                "Todos os dados locais e calibrações foram removidos com sucesso."
            )
            logger.info("Todos os dados do usuário foram limpos a pedido do usuário.")

    def _hotkey_toggle_pause(self):
        currently_paused = self.state_machine.state == AppState.PAUSED
        new_paused = not currently_paused
        self.toggle_pause(new_paused)
        if self.control_panel:
            self.root.after(
                0, lambda: self.control_panel.update_pause_text(new_paused)
            )

    def toggle_pause(self, paused: bool):
        if paused:
            success = self.state_machine.try_transition(AppState.PAUSED)
            if success:
                logger.info("Aplicação pausada.")
                self.dwell_clicker.reset()
                self.scroll_controller.deactivate()
                if self.action_bar:
                    self.action_bar.set_paused(True)
                    self.action_bar.set_scrolling(False)
        else:
            success = self.state_machine.try_transition(AppState.ACTIVE)
            if success:
                logger.info("Aplicação retomada.")
                self.mouse_controller.reset_smoothing()
                self.validator.reset()
                self.dwell_clicker.reset()
                if self.action_bar:
                    self.action_bar.set_paused(False)

    @property
    def is_paused(self) -> bool:
        return self.state_machine.state == AppState.PAUSED

    @property
    def is_calibrating(self) -> bool:
        return self.state_machine.state == AppState.CALIBRATING

    # ------------------------------------------------------------------
    # Acesso a dados thread-safe
    # ------------------------------------------------------------------

    def get_latest_gaze_raw(self) -> Optional[np.ndarray]:
        with self.data_lock:
            return self.latest_gaze_raw

    def get_latest_gaze_timestamp(self) -> float:
        with self.data_lock:
            return self.latest_gaze_timestamp

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.data_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_latest_valid_observation(self) -> Optional[TrackingResult]:
        with self.data_lock:
            return self.latest_valid_observation

    def get_latest_preview_frame(self) -> Optional[np.ndarray]:
        with self.data_lock:
            if self.latest_preview_frame is not None:
                return self.latest_preview_frame.copy()
            return None

    def update_smoothing(self, value: float):
        self.mouse_controller.set_smoothing_alpha(value)

    def quit_app(self):
        logger.info("Encerrando aplicação...")
        self.running = False

        if FT_SAFETY_RELEASE:
            self.mouse_controller.release_all()

        if self.action_bar:
            try:
                self.action_bar.destroy()
            except Exception:
                pass

        if self.setup_wizard and hasattr(self.setup_wizard, "window"):
            try:
                self.setup_wizard.window.destroy()
            except Exception:
                pass

        if self.training_ui and hasattr(self.training_ui, "window"):
            try:
                self.training_ui.window.destroy()
            except Exception:
                pass

        self.state_machine.try_transition(AppState.SHUTTING_DOWN)

        try:
            keyboard.unhook_all()
        except Exception:
            pass

        if self.camera_capture:
            self.camera_capture.release()
        elif self.cap and self.cap.isOpened():
            self.cap.release()

        # Fechar explicitamente o MediaPipe
        self.gaze_tracker.close()

        report = self.profiler.report()
        logger.info("Relatório de benchmark:\n%s", report)

        self.root.quit()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Fila da UI (latest-wins)
    # ------------------------------------------------------------------

    def _post_ui_update(self, key: str, value: Any) -> None:
        with self._ui_lock:
            self._ui_updates[key] = value

    def update_ui_loop(self):
        with self._ui_lock:
            updates = dict(self._ui_updates)
            self._ui_updates.clear()

        if self.control_panel:
            fps = updates.get("fps", int(self.profiler.capture_fps))
            left_ear = updates.get("left_ear", 0.0)
            right_ear = updates.get("right_ear", 0.0)
            threshold = updates.get("threshold", 0.2)

            if any(k in updates for k in ("fps", "left_ear", "right_ear", "threshold")):
                self.control_panel.update_status(fps, left_ear, right_ear, threshold)

            if "face_detected" in updates:
                face_det = updates["face_detected"]
                app_state = self.state_machine.state
                if hasattr(self.control_panel, 'update_face_status'):
                    self.control_panel.update_face_status(face_det, app_state.name)

            if hasattr(self.control_panel, "update_metrics"):
                latency_ms = self.profiler.stats(FrameProfiler.STAGE_TOTAL).mean_ms
                holdout_err = getattr(self.calibration_manager, "last_holdout_error", 0.0)
                is_drag = getattr(self.mouse_controller, "is_dragging", False)
                is_prec = getattr(self.mouse_controller, "is_precision_mode", False)
                prof_name = getattr(self.profile_manager, "active_profile_name", "hybrid")
                self.control_panel.update_metrics(
                    fps=fps,
                    process_fps=self.profiler.process_fps,
                    holdout_error=holdout_err,
                    latency_ms=latency_ms,
                    is_dragging=is_drag,
                    is_precision=is_prec,
                    profile_name=prof_name,
                )

        if self.running:
            self.root.after(100, self.update_ui_loop)

    # ------------------------------------------------------------------
    # Thread da Câmera (produtora)
    # ------------------------------------------------------------------

    def camera_loop(self):
        consecutive_failures = 0
        MAX_FAILURES = 30

        while self.running:
            if self.camera_capture:
                # Utiliza CameraCapture com buffer mínimo
                packet = self.camera_capture.get_latest_frame(timeout=0.1)
                if packet is None:
                    continue

                self.profiler.record_capture_fps()
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.profiler.record_dropped_frame()
                    except queue.Empty:
                        pass
                self.frame_queue.put(packet)
            else:
                # Fallback legado usando cap.read()
                t0 = time.perf_counter_ns()
                ret, frame = self.cap.read()
                elapsed_ns = time.perf_counter_ns() - t0

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_FAILURES:
                        logger.error("Câmera desconectada ou erro de leitura.")
                        if FT_SAFETY_RELEASE:
                            self.mouse_controller.release_all()
                        self.state_machine.try_transition(AppState.ERROR)
                        self.root.after(
                            0,
                            lambda: messagebox.showerror(
                                "Erro de Câmera",
                                "A câmera foi desconectada.\nVerifique a conexão e reinicie o aplicativo.",
                            ),
                        )
                        self.running = False
                        break
                    time.sleep(0.033)
                    continue

                consecutive_failures = 0
                self.profiler._record(FrameProfiler.STAGE_CAPTURE, elapsed_ns)
                self.profiler.record_capture_fps()

                self._legacy_frame_counter += 1
                packet = FramePacket(
                    frame_id=self._legacy_frame_counter,
                    capture_timestamp=time.perf_counter(),
                    image=frame,
                    camera_metadata={},
                )

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.profiler.record_dropped_frame()
                    except queue.Empty:
                        pass
                self.frame_queue.put(packet)

    # ------------------------------------------------------------------
    # Thread de Processamento (consumidora)
    # ------------------------------------------------------------------

    def processing_loop(self):
        while self.running:
            try:
                frame_item = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t_frame_start = time.perf_counter_ns()
            current_time = time.time()

            if isinstance(frame_item, FramePacket):
                packet = frame_item
            else:
                self._legacy_frame_counter += 1
                packet = FramePacket(
                    frame_id=self._legacy_frame_counter,
                    capture_timestamp=time.perf_counter(),
                    image=frame_item,
                    camera_metadata={},
                )

            # 1. Inferência MediaPipe (modo VIDEO com timestamp monotônico)
            with self.profiler.measure(FrameProfiler.STAGE_INFERENCE):
                raw_result = self.gaze_tracker.process_frame_packet(packet)

            # 2. Validação temporal e período de estabilização pós-perda
            if FT_TRACKING_VALIDATOR:
                result = self.validator.validate(raw_result)
            else:
                result = raw_result

            face_detected = result.tracking_valid
            landmarks = result.landmarks

            # 3. Gerenciamento do preview visual desacoplado
            if landmarks and DEBUG_DRAW:
                self.gaze_tracker.draw_debug(packet.image, landmarks)

            now_perf = time.perf_counter()
            if SHOW_PREVIEW and (now_perf - self._last_preview_render_time >= (1.0 / max(PREVIEW_FPS, 1))):
                self._last_preview_render_time = now_perf
                with self.data_lock:
                    self.latest_frame = packet.image.copy()
                    self.latest_preview_frame = packet.image.copy()

            # 4. Transição de estado ao perder ou recuperar rosto
            if face_detected:
                self.last_face_time = current_time
                if (FT_STATE_MACHINE and self.state_machine.state == AppState.TRACKING_LOST):
                    self.state_machine.try_transition(AppState.ACTIVE)
                    self.mouse_controller.reset_smoothing()
                    self.dwell_clicker.reset()
            else:
                time_since_face = current_time - self.last_face_time
                if (FT_STATE_MACHINE and
                        self.state_machine.state == AppState.ACTIVE and
                        time_since_face > TRACKING_LOST_TIMEOUT_SEC):
                    self.state_machine.try_transition(AppState.TRACKING_LOST)
                    self.dwell_clicker.reset()
                    if self.mouse_controller.is_dragging:
                        self.mouse_controller.stop_drag()
                        if self.action_bar:
                            self.action_bar.set_dragging(False)

            # 5. Controle do mouse — apenas se rastreamento for válido, NÃO expirado e estabilizado
            if face_detected:
                avg_iris = result.features.get("avg_iris")
                with self.data_lock:
                    self.latest_gaze_raw = avg_iris
                    self.latest_gaze_timestamp = current_time
                    self.latest_valid_observation = result

                mouse_allowed = (
                    self.state_machine.mouse_allowed
                    if FT_STATE_MACHINE
                    else (not self.is_paused and not self.is_calibrating)
                )

                # Bloqueio estrito: frames expirados ou em estabilização NÃO movem nem clicam
                can_act = mouse_allowed and result.is_stabilized and not result.is_expired(MAX_OBSERVATION_AGE_SEC)

                if can_act:
                    with self.profiler.measure(FrameProfiler.STAGE_MAPPING):
                        screen_pos = self.calibration_manager.map_to_screen(avg_iris)

                    if screen_pos and not self.blink_detector.is_calibrating:
                        sx, sy = screen_pos

                        # M5.2: Click Freeze e buffer de coordenadas pré-oclusão
                        if FT_GESTURE_ENGINE:
                            self.gesture_engine.record_stable_position(sx, sy, timestamp=current_time)
                            sx, sy = self.gesture_engine.get_stabilized_position((sx, sy), timestamp=current_time)

                        # M5.7: Modo Rolagem Dedicado
                        if FT_SCROLL_MODE and self.scroll_controller.is_active:
                            scroll_delta = self.scroll_controller.update(sy, timestamp=current_time)
                            if scroll_delta and not BENCHMARK_MODE:
                                self.mouse_controller.scroll(scroll_delta)

                        # Movimento do cursor
                        with self.profiler.measure(FrameProfiler.STAGE_MOUSE):
                            if not BENCHMARK_MODE:
                                self.mouse_controller.move(sx, sy, timestamp=current_time)

                        # M5.4: Dwell Click (clique por fixação com proteção anti-loop)
                        if FT_DWELL_CLICK and self.profile_manager.active_config.enable_dwell_click:
                            dwell_trig, dwell_prog, dwell_coord = self.dwell_clicker.update(
                                sx, sy, timestamp=current_time
                            )
                            self._post_ui_update("dwell_progress", dwell_prog)

                            if dwell_trig and not BENCHMARK_MODE:
                                consumed = False
                                if self.action_bar and self.action_bar.is_visible():
                                    consumed = self.action_bar.handle_dwell_click(int(sx), int(sy))

                                if not consumed:
                                    action = MouseAction.LEFT_CLICK
                                    if self.action_bar:
                                        action = self.action_bar.consume_armed_action()
                                    self._execute_mouse_action(action)

                    # Detecção e injeção de piscadas/gestos
                    img_h, img_w = packet.image.shape[:2]
                    l_blink, r_blink, d_blink, hold_start, hold_end, ears = (
                        self.blink_detector.process(landmarks, img_w, img_h)
                    )

                    if FT_GESTURE_ENGINE:
                        cfg = self.profile_manager.active_config
                        eff_l = l_blink if cfg.enable_left_blink else False
                        eff_r = r_blink if cfg.enable_right_blink else False
                        eff_d = d_blink if cfg.enable_double_blink else False
                        eff_hs = hold_start if cfg.enable_hold_drag else False
                        eff_he = hold_end if cfg.enable_hold_drag else False

                        action = self.gesture_engine.process_blink_events(
                            eff_l, eff_r, eff_d, eff_hs, eff_he, timestamp=current_time
                        )

                        if action and not BENCHMARK_MODE:
                            if action in (MouseAction.LEFT_CLICK, MouseAction.RIGHT_CLICK, MouseAction.DOUBLE_CLICK):
                                if self.action_bar:
                                    armed = self.action_bar.consume_armed_action()
                                    if armed != MouseAction.LEFT_CLICK:
                                        action = armed
                            self._execute_mouse_action(action)
                    else:
                        if not BENCHMARK_MODE:
                            if l_blink:
                                self.mouse_controller.left_click()
                            if r_blink:
                                self.mouse_controller.right_click()
                            if d_blink:
                                self.mouse_controller.double_click()
                            if hold_start:
                                self.mouse_controller.start_drag()
                            if hold_end:
                                self.mouse_controller.stop_drag()
                else:
                    ears = (0.0, 0.0)

                current_thresh = self.blink_detector.ear_threshold
                self._post_ui_update("left_ear", ears[0] if len(ears) > 0 else 0.0)
                self._post_ui_update("right_ear", ears[1] if len(ears) > 1 else 0.0)
                self._post_ui_update("threshold", current_thresh)
            else:
                with self.data_lock:
                    self.latest_valid_observation = None

            t_frame_total = time.perf_counter_ns() - t_frame_start
            self.profiler._record(FrameProfiler.STAGE_TOTAL, t_frame_total)
            self.profiler.record_process_fps(face_detected=face_detected)
            self.profiler.record_observation_age(current_time - self.last_face_time)

            capture_fps = self.profiler.capture_fps
            self._post_ui_update("fps", int(capture_fps))
            self._post_ui_update("face_detected", face_detected)


if __name__ == "__main__":
    app = EyeMouseApp()
    app.root.mainloop()
