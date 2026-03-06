import cv2
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import logging
import sys
import queue

from gaze_tracker import GazeTracker
from blink_detector import BlinkDetector
from calibration import CalibrationManager
from mouse_controller import MouseController
from ui.calibration_ui import CalibrationUI
from ui.control_panel import ControlPanel
from config import *

# Configuração de Log
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class EyeMouseApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Esconder janela principal raiz

        # Solicitar perfil de usuário (Melhoria 15)
        self.user_profile = simpledialog.askstring(
            "Perfil de Usuário", 
            "Digite seu nome (ou deixe em branco para 'default'):", 
            parent=self.root
        )
        if not self.user_profile:
            self.user_profile = "default"

        self.running = False
        self.is_paused = False
        self.is_calibrating = False
        self.calibration_ui = None
        self.control_panel = None

        # Estado compartilhado (Thread-Safe - Melhoria 16)
        self.data_lock = threading.Lock()
        self.latest_gaze_raw = None  # (x, y) normalizado
        self.latest_frame = None  # Frame da câmera com anotações
        self.last_face_time = 0
        self.fps = 0
        
        # Fila de frames (Câmera -> Processamento - Melhoria 17)
        # Maxsize pequeno para garantir baixa latência (drop frame se processamento lento)
        self.frame_queue = queue.Queue(maxsize=1)

        # Inicializar módulos
        try:
            self.gaze_tracker = GazeTracker()
            self.blink_detector = BlinkDetector()
            # Inicializar com perfil selecionado
            self.calibration_manager = CalibrationManager(profile_name=self.user_profile)
            self.mouse_controller = MouseController()
            logging.info(f"Módulos inicializados com sucesso. Perfil: {self.user_profile}")
        except Exception as e:
            logging.error(f"Erro ao inicializar módulos: {e}")
            messagebox.showerror("Erro Fatal", f"Falha ao iniciar: {e}")
            sys.exit(1)

        # Verificar Câmera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            logging.error("Câmera não encontrada.")
            messagebox.showerror(
                "Erro de Câmera",
                f"Não foi possível acessar a câmera (Index {CAMERA_INDEX}). Verifique a conexão.",
            )
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Verificar Calibração
        if self.calibration_manager.load_calibration():
            use_calib = messagebox.askyesno(
                "Calibração Encontrada", 
                f"Calibração encontrada para '{self.user_profile}'. Deseja usar?"
            )
            if not use_calib:
                self.start_calibration()
            else:
                self.show_control_panel()
        else:
            self.start_calibration()

        # Iniciar threads (Melhoria 17)
        self.running = True
        
        # Thread produtora (Câmera)
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Thread consumidora (Processamento)
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # Loop de atualização da UI (Status)
        self.root.after(500, self.update_ui_loop)

    def start_calibration(self):
        self.is_calibrating = True
        self.is_paused = True  # Pausa controle do mouse
        logging.info("Iniciando calibração...")

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
        """Inicia a calibração de piscada (Melhoria 6)."""
        self.blink_detector.start_calibration(duration=10.0)
        messagebox.showinfo(
            "Calibração de Piscada", 
            "Olhe para a tela e pisque normalmente por 10 segundos.\n"
            "O sistema ajustará a sensibilidade automaticamente."
        )

    def on_calibration_complete(self, cancelled=False):
        if cancelled:
            self.is_calibrating = False
            self.is_paused = False
            logging.info("Calibração cancelada pelo usuário.")
            if self.control_panel:
                self.control_panel.window.deiconify()
            else:
                self.show_control_panel()
            return

        # Calcular e validar calibração (Melhoria 12)
        success, error = self.calibration_manager.compute_calibration()
        
        if success:
            if error > CALIBRATION_REPROJECTION_ERROR_THRESHOLD:
                retry = messagebox.askyesno(
                    "Calibração Imprecisa",
                    f"A qualidade da calibração está baixa (Erro: {error:.1f}px).\n"
                    f"Recomendado: < {CALIBRATION_REPROJECTION_ERROR_THRESHOLD}px.\n\n"
                    "Deseja tentar novamente?"
                )
                if retry:
                    # Reiniciar calibração
                    self.start_calibration()
                    return
            else:
                messagebox.showinfo(
                    "Sucesso", 
                    f"Calibração concluída com sucesso!\nErro médio: {error:.1f}px"
                )
        else:
            messagebox.showerror("Erro", "Falha ao computar calibração. Tente novamente.")
            self.start_calibration()
            return

        self.is_calibrating = False
        self.is_paused = False
        logging.info("Calibração concluída.")

        if self.control_panel:
            self.control_panel.window.deiconify()
        else:
            self.show_control_panel()

    def show_control_panel(self):
        if not self.control_panel:
            self.control_panel = ControlPanel(
                self.root,
                self.toggle_pause,
                self.start_calibration,
                self.quit_app,
                self.update_smoothing,
                self.start_blink_calibration # Novo callback
            )

    def get_latest_gaze_raw(self):
        with self.data_lock:
            return self.latest_gaze_raw

    def get_latest_frame(self):
        with self.data_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def toggle_pause(self, paused):
        self.is_paused = paused
        logging.info(f"Pausado: {paused}")

    def update_smoothing(self, value):
        self.mouse_controller.set_smoothing_alpha(value)

    def quit_app(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.quit()
        sys.exit(0)

    def update_ui_loop(self):
        if self.control_panel:
            # Obter estado atual dos olhos (simplificado aqui, ideal seria thread-safe)
            # Como é UI, não precisa ser perfeito em tempo real
            pass

        self.root.after(500, self.update_ui_loop)

    def camera_loop(self):
        """Thread produtora: Captura frames da câmera."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Melhoria 16: Fila thread-safe com drop de frames antigos
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def processing_loop(self):
        """Thread consumidora: Processa frames e controla mouse."""
        last_time = time.time()

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_time = time.time()

            # Melhoria 18: Reduzir resolução para processamento (320x240)
            # O MediaPipe e GazeTracker trabalham com coords normalizadas,
            # então não é necessário ajustar escala dos resultados.
            small_frame = cv2.resize(frame, (320, 240))

            # Processamento
            left_iris, right_iris, landmarks = self.gaze_tracker.process_frame(small_frame)

            if landmarks:
                # Desenhar debug no frame ORIGINAL (640x480)
                # Como landmarks são normalizados, draw_debug funciona em qualquer resolução
                self.gaze_tracker.draw_debug(frame, landmarks)
                with self.data_lock:
                    self.latest_frame = frame.copy()

            if left_iris is not None and right_iris is not None:
                self.last_face_time = current_time

                # Calcular ponto médio das íris (raw)
                avg_iris = (left_iris + right_iris) / 2.0
                
                with self.data_lock:
                    self.latest_gaze_raw = avg_iris

                # Se estiver calibrando, o UI coleta os dados via get_latest_gaze_raw
                # Se NÃO estiver calibrando e NÃO estiver pausado, controla o mouse
                if not self.is_calibrating and not self.is_paused:

                    # 1. Mapear para tela
                    screen_pos = self.calibration_manager.map_to_screen(avg_iris)

                    # Mover apenas se não estiver calibrando piscada (para evitar distração)
                    if screen_pos and not self.blink_detector.is_calibrating:
                        sx, sy = screen_pos
                        self.mouse_controller.move(sx, sy)

                    # 2. Detectar Piscadas
                    img_h, img_w = frame.shape[:2]
                    l_blink, r_blink, d_blink, hold_start, hold_end, ears = (
                        self.blink_detector.process(landmarks, img_w, img_h)
                    )

                    # Ações do Mouse
                    if l_blink:
                        self.mouse_controller.click("left")
                    if r_blink:
                        self.mouse_controller.click("right")
                    if d_blink:
                        self.mouse_controller.double_click()
                    
                    if hold_start:
                        self.mouse_controller.start_drag()
                    if hold_end:
                        self.mouse_controller.stop_drag()
                        
                    # Atualizar UI
                    fps = int(1.0 / (current_time - last_time)) if current_time > last_time else 0
                    if self.control_panel:
                        # Passar threshold atual para UI
                        current_thresh = self.blink_detector.ear_threshold
                        self.root.after(
                            0,
                            lambda f=fps, l=ears[0], r=ears[1], t=current_thresh: self.control_panel.update_status(
                                f, l, r, t
                            ),
                        )

            last_time = current_time

if __name__ == "__main__":
    app = EyeMouseApp()
    app.root.mainloop()
