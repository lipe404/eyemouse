import cv2
import threading
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import logging
import sys

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
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EyeMouseApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw() # Esconder janela principal raiz
        
        self.running = False
        self.is_paused = False
        self.is_calibrating = False
        self.calibration_ui = None
        self.control_panel = None
        
        # Estado compartilhado
        self.latest_gaze_raw = None # (x, y) normalizado
        self.last_face_time = 0
        self.fps = 0
        
        # Inicializar módulos
        try:
            self.gaze_tracker = GazeTracker()
            self.blink_detector = BlinkDetector()
            self.calibration_manager = CalibrationManager()
            self.mouse_controller = MouseController()
            logging.info("Módulos inicializados com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao inicializar módulos: {e}")
            messagebox.showerror("Erro Fatal", f"Falha ao iniciar: {e}")
            sys.exit(1)

        # Verificar Câmera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            logging.error("Câmera não encontrada.")
            messagebox.showerror("Erro de Câmera", f"Não foi possível acessar a câmera (Index {CAMERA_INDEX}). Verifique a conexão.")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Verificar Calibração
        if self.calibration_manager.load_calibration():
            use_calib = messagebox.askyesno("Calibração Encontrada", "Deseja usar a calibração salva?")
            if not use_calib:
                self.start_calibration()
            else:
                self.show_control_panel()
        else:
            self.start_calibration()
            
        # Iniciar thread de rastreamento
        self.running = True
        self.tracker_thread = threading.Thread(target=self.tracking_loop, daemon=True)
        self.tracker_thread.start()
        
        # Loop de atualização da UI (Status)
        self.root.after(500, self.update_ui_loop)

    def start_calibration(self):
        self.is_calibrating = True
        self.is_paused = True # Pausa controle do mouse
        logging.info("Iniciando calibração...")
        
        if self.control_panel:
            self.control_panel.window.withdraw()
            
        self.calibration_ui = CalibrationUI(
            self.root, 
            self.calibration_manager, 
            self.get_latest_gaze_raw,
            self.on_calibration_complete
        )

    def on_calibration_complete(self):
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
                self.update_smoothing
            )

    def get_latest_gaze_raw(self):
        return self.latest_gaze_raw

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

    def tracking_loop(self):
        last_time = time.time()
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            current_time = time.time()
            
            # Processamento
            left_iris, right_iris, landmarks = self.gaze_tracker.process_frame(frame)
            
            if left_iris is not None and right_iris is not None:
                self.last_face_time = current_time
                
                # Calcular ponto médio das íris (raw)
                avg_iris = (left_iris + right_iris) / 2.0
                self.latest_gaze_raw = avg_iris
                
                # Se estiver calibrando, o UI coleta os dados via get_latest_gaze_raw
                # Se NÃO estiver calibrando e NÃO estiver pausado, controla o mouse
                if not self.is_calibrating and not self.is_paused:
                    
                    # 1. Mapear para tela
                    screen_pos = self.calibration_manager.map_to_screen(avg_iris)
                    
                    if screen_pos:
                        sx, sy = screen_pos
                        self.mouse_controller.move(sx, sy)
                    
                    # 2. Detectar Piscadas
                    img_h, img_w = frame.shape[:2]
                    l_blink, r_blink, d_blink, hold_start, hold_end, ears = self.blink_detector.process(landmarks, img_w, img_h)
                    
                    # Ações
                    if d_blink:
                        logging.info("Duplo Clique")
                        self.mouse_controller.double_click()
                    elif l_blink:
                        logging.info("Clique Esquerdo")
                        self.mouse_controller.click('left')
                    elif r_blink:
                        logging.info("Clique Direito")
                        self.mouse_controller.click('right')
                        
                    if hold_start:
                        logging.info("Iniciar Arraste")
                        self.mouse_controller.start_drag()
                    elif hold_end:
                        logging.info("Soltar Arraste")
                        self.mouse_controller.stop_drag()

                    # Atualizar UI status (FPS e Olhos)
                    if self.control_panel and frame_count % 5 == 0: # Atualizar a cada 5 frames para não travar
                         # Calcular FPS
                        dt = current_time - last_time
                        if dt > 0:
                            fps = 1.0 / dt
                            # Ears
                            l_open = ears[0] > BLINK_EAR_THRESHOLD
                            r_open = ears[1] > BLINK_EAR_THRESHOLD
                            
                            # Executar na thread principal
                            self.root.after(0, lambda f=fps, l=l_open, r=r_open: self.control_panel.update_status(f, l, r))

            else:
                # Face perdida
                if current_time - self.last_face_time > 2.0:
                    if not self.is_paused and not self.is_calibrating:
                        # Auto-pause por segurança
                        # Mas cuidado para não conflitar com UI thread
                        # Apenas logar por enquanto ou mostrar aviso
                        pass

            last_time = current_time
            frame_count += 1
            
            # Manter framerate alvo
            elapsed = time.time() - current_time
            wait = max(1, int((1.0/TARGET_FPS - elapsed) * 1000))
            time.sleep(wait / 1000.0)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EyeMouseApp()
    app.run()
