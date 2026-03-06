import tkinter as tk
import time
from threading import Thread
import numpy as np
from config import SCREEN_MARGIN, CALIBRATION_FRAMES_PER_POINT


class CalibrationUI:
    def __init__(self, root, calibration_manager, get_latest_gaze_fn, on_complete):
        self.root = root
        self.calib_manager = calibration_manager
        self.get_latest_gaze = get_latest_gaze_fn
        self.on_complete = on_complete

        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()

        # Configurar janela fullscreen
        self.window = tk.Toplevel(root)
        self.window.title("Calibração EyeMouse")
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="black")
        self.window.focus_force()

        # Canvas para desenhar pontos
        self.canvas = tk.Canvas(
            self.window,
            width=self.width,
            height=self.height,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Pontos de calibração (grade 3x3 com margem)
        margin = SCREEN_MARGIN
        w, h = self.width, self.height
        cx, cy = w // 2, h // 2

        self.points = [
            (margin, margin),
            (cx, margin),
            (w - margin, margin),
            (margin, cy),
            (cx, cy),
            (w - margin, cy),
            (margin, h - margin),
            (cx, h - margin),
            (w - margin, h - margin),
        ]

        self.current_point_idx = 0
        self.frames_collected = 0
        self.max_frames = CALIBRATION_FRAMES_PER_POINT
        self.is_collecting = False

        # Elementos da UI
        self.instruction = self.canvas.create_text(
            cx,
            cy - 100,
            text="Olhe para o ponto vermelho",
            fill="white",
            font=("Arial", 24),
        )
        self.progress_bar_bg = self.canvas.create_rectangle(
            cx - 100, cy + 100, cx + 100, cy + 120, outline="white"
        )
        self.progress_bar_fill = self.canvas.create_rectangle(
            cx - 100, cy + 100, cx - 100, cy + 120, fill="green", outline=""
        )

        # Iniciar após breve delay
        self.window.after(1000, self.start_sequence)

    def start_sequence(self):
        self.show_point()

    def show_point(self):
        if self.current_point_idx >= len(self.points):
            self.finish_calibration()
            return

        x, y = self.points[self.current_point_idx]

        # Limpar canvas (exceto textos fixos se houver)
        self.canvas.delete("target")

        # Desenhar alvo
        r = 20
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="red", outline="white", tags="target"
        )
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", tags="target")

        # Resetar contadores
        self.frames_collected = 0
        self.update_progress(0)

        # Aguardar um pouco para o usuário focar (500ms)
        self.window.after(500, self.start_collection)

    def start_collection(self):
        self.is_collecting = True
        self.collect_loop()

    def collect_loop(self):
        if not self.is_collecting:
            return

        if self.frames_collected >= self.max_frames:
            self.is_collecting = False
            self.current_point_idx += 1
            self.show_point()
            return

        # Obter dados do gaze (média dos dois olhos se possível)
        gaze_data = self.get_latest_gaze()

        if gaze_data is not None:
            # gaze_data deve ser (left_iris, right_iris) ou similar
            # Vamos assumir que get_latest_gaze retorna um array numpy ou tupla (x, y) combinado
            # Se retornar None, ignoramos este frame

            # Adicionar ao gerenciador de calibração
            # Precisamos normalizar o gaze_data? O CalibrationManager espera o raw iris coords.
            # E espera o screen coord correspondente.

            # Recuperar coordenadas da tela do ponto atual
            screen_x, screen_y = self.points[self.current_point_idx]

            self.calib_manager.add_point(gaze_data, (screen_x, screen_y))

            self.frames_collected += 1
            self.update_progress(self.frames_collected / self.max_frames)

        # Tentar novamente em breve (aprox 30 FPS = 33ms)
        self.window.after(33, self.collect_loop)

    def update_progress(self, percent):
        cx = self.width // 2
        cy = self.height // 2
        bar_width = 200
        filled_width = int(bar_width * percent)
        self.canvas.coords(
            self.progress_bar_fill,
            cx - 100,
            cy + 100,
            cx - 100 + filled_width,
            cy + 120,
        )

    def finish_calibration(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.width // 2,
            self.height // 2,
            text="Calibrando...",
            fill="white",
            font=("Arial", 24),
        )
        self.window.update()

        success = self.calib_manager.compute_calibration()

        if success:
            self.canvas.create_text(
                self.width // 2,
                self.height // 2 + 50,
                text="Sucesso!",
                fill="green",
                font=("Arial", 20),
            )
        else:
            self.canvas.create_text(
                self.width // 2,
                self.height // 2 + 50,
                text="Falha na calibração",
                fill="red",
                font=("Arial", 20),
            )

        self.window.after(1000, self.close)

    def close(self):
        self.window.destroy()
        if self.on_complete:
            self.on_complete()
