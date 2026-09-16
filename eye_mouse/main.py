"""
main.py — Ponto de entrada e orquestrador do EyeMouse.

Mudanças do Milestone 1:
  - AppState machine integrada (estados explícitos, transições validadas).
  - release_all() chamado em TODAS as transições de estado que encerram
    o modo ativo (pausa, erro, encerramento, perda de câmera).
  - Fila de atualizações de UI (latest-wins dict): sem acumulação de
    chamadas root.after() do loop de processamento.
  - Benchmark integrado (FrameProfiler) — coleta métricas por etapa.
  - BENCHMARK_MODE: roda pipeline sem mover cursor real.
  - Validação de perfil (CalibrationManager agora valida nomes).
  - Detecção de TRACKING_LOST com timeout configurável.
"""
import cv2
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import logging
import sys
import queue
try:
    import keyboard
except ImportError:
    # Stub para ambientes de teste sem a biblioteca keyboard instalada
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
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    LOG_FILE, CALIBRATION_REPROJECTION_ERROR_THRESHOLD,
    TRACKING_LOST_TIMEOUT_SEC, BENCHMARK_MODE,
    FT_STATE_MACHINE, FT_SAFETY_RELEASE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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

    Gerencia o ciclo de vida, threads de câmera e processamento,
    interface de usuário e coordenação entre os módulos.
    """

    def __init__(self):
        """Inicializa a aplicação."""
        self.root = tk.Tk()
        self.root.withdraw()

        # --- Perfil de usuário ---
        raw_profile = simpledialog.askstring(
            "Perfil de Usuário",
            "Digite seu nome (ou deixe em branco para 'default'):",
            parent=self.root,
        )
        self.user_profile = raw_profile.strip() if raw_profile and raw_profile.strip() else "default"
        # Sanitizar: aceitar apenas caracteres válidos; fallback para 'default'
        import re
        if not re.match(r'^[A-Za-z0-9_\-]{1,64}$', self.user_profile):
            logger.warning("Perfil '%s' inválido; usando 'default'.", self.user_profile)
            self.user_profile = "default"

        # --- Estado compartilhado (thread-safe) ---
        self.data_lock = threading.Lock()
        self.latest_gaze_raw = None          # (x, y) normalizado
        self.latest_gaze_timestamp: float = 0.0   # timestamp da última atualização
        self.latest_frame = None             # Frame anotado para CalibrationUI
        self.last_face_time: float = 0.0     # Timestamp da última detecção de rosto

        # --- Fila de frames (câmera → processamento) ---
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

        # --- Fila de atualizações da UI (latest-wins dict) ---
        # O loop de processamento escreve; o update_ui_loop lê na thread principal.
        self._ui_updates: dict = {}
        self._ui_lock = threading.Lock()

        # --- Controle de execução ---
        self.running: bool = False
        self.calibration_ui = None
        self.control_panel = None

        # --- Benchmark ---
        self.profiler = FrameProfiler()

        # --- Hotkey global ---
        try:
            keyboard.add_hotkey("ctrl+shift+p", self._hotkey_toggle_pause)
        except Exception as exc:
            logger.error("Erro ao registrar hotkey: %s", exc)

        # --- Módulos ---
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

        # --- Máquina de estados ---
        # on_release_all é chamado automaticamente em transições críticas.
        self.state_machine = StateMachine(
            initial=AppState.INITIALIZING,
            on_release_all=self.mouse_controller.release_all if FT_SAFETY_RELEASE else None,
        )

        # --- Câmera ---
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
            logger.info(
                "BENCHMARK_MODE ativo: pipeline roda mas cursor NÃO será movido."
            )

        # --- Iniciar fluxo de calibração ou controle ---
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

        # --- Iniciar threads ---
        self.running = True
        self.camera_thread = threading.Thread(
            target=self.camera_loop, daemon=True, name="camera"
        )
        self.camera_thread.start()

        self.processing_thread = threading.Thread(
            target=self.processing_loop, daemon=True, name="processing"
        )
        self.processing_thread.start()

        # Loop de atualização da UI (100 ms)
        self.root.after(100, self.update_ui_loop)

    # ------------------------------------------------------------------
    # Calibração
    # ------------------------------------------------------------------

    def _enter_active_after_calibration_load(self):
        """Transita para ACTIVE após carregar calibração existente."""
        try:
            self.state_machine.transition(AppState.CALIBRATING)
            self.state_machine.transition(AppState.ACTIVE)
        except InvalidTransition:
            pass
        self.show_control_panel()

    def start_calibration(self):
        """Inicia o processo de calibração de tela."""
        logger.info("Iniciando calibração...")
        self.state_machine.try_transition(AppState.CALIBRATING)

        if self.control_panel:
            self.control_panel.window.withdraw()

        self.calibration_ui = CalibrationUI(
            self.root,
            self.calibration_manager,
            self.get_latest_gaze_raw,
            self.on_calibration_complete,
            self.get_latest_frame,
        )

    def start_blink_calibration(self):
        """Inicia a calibração automática de piscada."""
        self.blink_detector.start_calibration(duration=10.0)
        messagebox.showinfo(
            "Calibração de Piscada",
            "Olhe para a tela e pisque normalmente por 10 segundos.\n"
            "O sistema ajustará a sensibilidade automaticamente.",
        )

    def on_calibration_complete(self, cancelled: bool = False):
        """
        Callback chamado ao finalizar a calibração.

        Args:
            cancelled: True se o usuário cancelou.
        """
        if cancelled:
            logger.info("Calibração cancelada.")
            self.state_machine.try_transition(AppState.ACTIVE)
            if self.control_panel:
                self.control_panel.window.deiconify()
            else:
                self.show_control_panel()
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

    # ------------------------------------------------------------------
    # Painel de controle
    # ------------------------------------------------------------------

    def show_control_panel(self):
        """Exibe o painel de controle flutuante."""
        if not self.control_panel:
            self.control_panel = ControlPanel(
                self.root,
                self.toggle_pause,
                self.start_calibration,
                self.quit_app,
                self.update_smoothing,
                self.start_blink_calibration,
            )

    # ------------------------------------------------------------------
    # Pausa
    # ------------------------------------------------------------------

    def _hotkey_toggle_pause(self):
        """Callback para Ctrl+Shift+P (thread da biblioteca keyboard)."""
        currently_paused = self.state_machine.state == AppState.PAUSED
        new_paused = not currently_paused
        self.toggle_pause(new_paused)
        if self.control_panel:
            self.root.after(
                0, lambda: self.control_panel.update_pause_text(new_paused)
            )

    def toggle_pause(self, paused: bool):
        """
        Alterna o estado de pausa.

        Em qualquer transição para PAUSED, release_all() é chamado
        automaticamente pelo StateMachine.

        Args:
            paused: True para pausar, False para retomar.
        """
        if paused:
            success = self.state_machine.try_transition(AppState.PAUSED)
            if success:
                logger.info("Aplicação pausada.")
        else:
            success = self.state_machine.try_transition(AppState.ACTIVE)
            if success:
                logger.info("Aplicação retomada.")
                # Reiniciar filtro de suavização para evitar salto de cursor
                self.mouse_controller.reset_smoothing()

    # Mantido para compatibilidade com ControlPanel (is_paused como bool)
    @property
    def is_paused(self) -> bool:
        return self.state_machine.state == AppState.PAUSED

    @property
    def is_calibrating(self) -> bool:
        return self.state_machine.state == AppState.CALIBRATING

    # ------------------------------------------------------------------
    # Acesso a dados thread-safe
    # ------------------------------------------------------------------

    def get_latest_gaze_raw(self):
        """Retorna a posição do olhar mais recente (thread-safe)."""
        with self.data_lock:
            return self.latest_gaze_raw

    def get_latest_gaze_timestamp(self) -> float:
        """Retorna o timestamp da última atualização de gaze (thread-safe)."""
        with self.data_lock:
            return self.latest_gaze_timestamp

    def get_latest_frame(self):
        """Retorna o frame mais recente (thread-safe, cópia)."""
        with self.data_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    # ------------------------------------------------------------------
    # Suavização
    # ------------------------------------------------------------------

    def update_smoothing(self, value):
        """Atualiza o fator de suavização do cursor."""
        self.mouse_controller.set_smoothing_alpha(value)

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------

    def quit_app(self):
        """
        Encerra a aplicação de forma segura.

        Garante que todos os botões do mouse sejam liberados antes de sair.
        """
        logger.info("Encerrando aplicação...")
        self.running = False

        # Liberar botões ANTES de qualquer outra ação
        if FT_SAFETY_RELEASE:
            self.mouse_controller.release_all()

        # Transitar para SHUTTING_DOWN (também chama release_all via StateMachine)
        self.state_machine.try_transition(AppState.SHUTTING_DOWN)

        try:
            keyboard.unhook_all()
        except Exception:
            pass

        if self.cap.isOpened():
            self.cap.release()

        # Logar relatório de benchmark antes de sair
        report = self.profiler.report()
        logger.info("Relatório de benchmark:\n%s", report)

        self.root.quit()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Fila de atualizações da UI
    # ------------------------------------------------------------------

    def _post_ui_update(self, key: str, value) -> None:
        """
        Publica uma atualização para a UI de forma thread-safe.

        Usa semântica "latest-wins": apenas o valor mais recente de
        cada chave é mantido, evitando acumulação de callbacks root.after().
        """
        with self._ui_lock:
            self._ui_updates[key] = value

    def update_ui_loop(self):
        """
        Consome atualizações da fila e aplica na UI (thread principal).

        Chamado via root.after(100) — sempre na thread principal do Tkinter.
        """
        with self._ui_lock:
            updates = dict(self._ui_updates)
            self._ui_updates.clear()

        if self.control_panel and updates:
            fps = updates.get("fps", 0)
            left_ear = updates.get("left_ear", 0.0)
            right_ear = updates.get("right_ear", 0.0)
            threshold = updates.get("threshold", 0.2)

            if any(k in updates for k in ("fps", "left_ear", "right_ear", "threshold")):
                self.control_panel.update_status(fps, left_ear, right_ear, threshold)

            if "face_detected" in updates:
                face_det = updates["face_detected"]
                app_state = self.state_machine.state
                status_text = (
                    f"Estado: {app_state.name} | "
                    f"Rosto: {'✓' if face_det else '✗'}"
                )
                # Tenta chamar update_face_status se o painel suportar
                if hasattr(self.control_panel, 'update_face_status'):
                    self.control_panel.update_face_status(face_det, app_state.name)

        if self.running:
            self.root.after(100, self.update_ui_loop)

    # ------------------------------------------------------------------
    # Thread da câmera (produtora)
    # ------------------------------------------------------------------

    def camera_loop(self):
        """Thread produtora: captura frames da câmera."""
        consecutive_failures = 0
        MAX_FAILURES = 30  # ~1 segundo a 30 FPS

        while self.running:
            t0 = time.perf_counter_ns()
            ret, frame = self.cap.read()
            elapsed_ns = time.perf_counter_ns() - t0

            if not ret:
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
                            "A câmera foi desconectada.\n"
                            "Verifique a conexão e reinicie o aplicativo.",
                        ),
                    )
                    self.running = False
                    break
                time.sleep(0.033)
                continue

            consecutive_failures = 0
            self.profiler._record(FrameProfiler.STAGE_CAPTURE, elapsed_ns)
            self.profiler.record_capture_fps()

            # Drop-on-full: manter latência baixa
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.profiler.record_dropped_frame()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    # ------------------------------------------------------------------
    # Thread de processamento (consumidora)
    # ------------------------------------------------------------------

    def processing_loop(self):
        """Thread consumidora: processa frames e controla o mouse."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t_frame_start = time.perf_counter_ns()
            current_time = time.time()

            # -- Pré-processamento --
            with self.profiler.measure(FrameProfiler.STAGE_PREPROCESS):
                small_frame = cv2.resize(frame, (320, 240))

            # -- Inferência MediaPipe --
            with self.profiler.measure(FrameProfiler.STAGE_INFERENCE):
                left_iris, right_iris, landmarks = self.gaze_tracker.process_frame(
                    small_frame
                )

            face_detected = left_iris is not None and right_iris is not None

            # -- Atualizar frame para CalibrationUI --
            if landmarks:
                self.gaze_tracker.draw_debug(frame, landmarks)
                with self.data_lock:
                    self.latest_frame = frame.copy()

            # -- Detecção de rosto perdido --
            if face_detected:
                self.last_face_time = current_time
                # Recuperar de TRACKING_LOST se rosto voltou
                if (FT_STATE_MACHINE and
                        self.state_machine.state == AppState.TRACKING_LOST):
                    self.state_machine.try_transition(AppState.ACTIVE)
                    self.mouse_controller.reset_smoothing()
            else:
                # Verificar timeout de rastreamento
                time_since_face = current_time - self.last_face_time
                if (FT_STATE_MACHINE and
                        self.state_machine.state == AppState.ACTIVE and
                        time_since_face > TRACKING_LOST_TIMEOUT_SEC):
                    self.state_machine.try_transition(AppState.TRACKING_LOST)

            # -- Extração de features e mapeamento --
            if face_detected:
                with self.profiler.measure(FrameProfiler.STAGE_FEATURES):
                    avg_iris = (left_iris + right_iris) / 2.0
                    with self.data_lock:
                        self.latest_gaze_raw = avg_iris
                        self.latest_gaze_timestamp = current_time

                # -- Controle do mouse (apenas no estado ACTIVE) --
                mouse_allowed = (
                    self.state_machine.mouse_allowed
                    if FT_STATE_MACHINE
                    else (not self.is_paused and not self.is_calibrating)
                )

                if mouse_allowed:
                    # Mapeamento gaze → tela
                    with self.profiler.measure(FrameProfiler.STAGE_MAPPING):
                        screen_pos = self.calibration_manager.map_to_screen(avg_iris)

                    # Suavização e movimento
                    if screen_pos and not self.blink_detector.is_calibrating:
                        sx, sy = screen_pos
                        with self.profiler.measure(FrameProfiler.STAGE_SMOOTHING):
                            # move() aplica o filtro internamente
                            pass

                        with self.profiler.measure(FrameProfiler.STAGE_MOUSE):
                            if not BENCHMARK_MODE:
                                self.mouse_controller.move(sx, sy)

                    # Detecção de piscadas
                    img_h, img_w = frame.shape[:2]
                    l_blink, r_blink, d_blink, hold_start, hold_end, ears = (
                        self.blink_detector.process(landmarks, img_w, img_h)
                    )

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

                # -- Publicar atualização de UI (sem root.after acumulativo) --
                current_thresh = self.blink_detector.ear_threshold
                self._post_ui_update("left_ear", ears[0] if len(ears) > 0 else 0.0)
                self._post_ui_update("right_ear", ears[1] if len(ears) > 1 else 0.0)
                self._post_ui_update("threshold", current_thresh)

            # -- FPS e métricas de frame --
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
