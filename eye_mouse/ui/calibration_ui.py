import tkinter as tk
import time
from threading import Thread
import numpy as np
import cv2
from config import SCREEN_MARGIN, CALIBRATION_FRAMES_PER_POINT, CALIBRATION_POINTS


class CalibrationUI:
    def __init__(
        self,
        root,
        calibration_manager,
        get_latest_gaze_fn,
        on_complete,
        get_latest_frame_fn=None,
    ):
        self.root = root
        self.calib_manager = calibration_manager
        self.get_latest_gaze = get_latest_gaze_fn
        self.on_complete = on_complete
        self.get_latest_frame = get_latest_frame_fn
        self.video_image_id = None
        self.tk_image = None

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

        # Pontos de calibração (grade NxN com margem) - Melhoria 13
        margin = SCREEN_MARGIN
        w, h = self.width, self.height
        
        # Determinar tamanho da grade (NxN)
        grid_size = int(np.sqrt(CALIBRATION_POINTS))
        if grid_size * grid_size != CALIBRATION_POINTS:
            grid_size = 4 # Fallback padrão 4x4
            
        self.points = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Calcular x, y distribuídos uniformemente
                x = margin + (j * (w - 2 * margin) / (grid_size - 1))
                y = margin + (i * (h - 2 * margin) / (grid_size - 1))
                self.points.append((int(x), int(y)))

        self.current_point_idx = 0
        self.frames_collected = 0
        self.max_frames = CALIBRATION_FRAMES_PER_POINT
        self.is_collecting = False
        self.animation_start_time = 0

        # Elementos da UI
        cx, cy = w // 2, h // 2
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

        # Handle window close (X button)
        self.window.protocol("WM_DELETE_WINDOW", self.on_user_close)

        # Iniciar após breve delay
        self.window.after(1000, self.start_sequence)

        # Iniciar loop de vídeo
        self.update_video_feed()

    def update_video_feed(self):
        if not self.window.winfo_exists():
            return

        if self.get_latest_frame:
            frame = self.get_latest_frame()
            if frame is not None:
                # Resize para preencher a tela
                frame_resized = cv2.resize(frame, (self.width, self.height))

                # Converter para RGB
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Codificar para PPM (suportado nativamente pelo Tkinter)
                success, buffer = cv2.imencode(".ppm", rgb)
                if success:
                    # Manter referência para evitar garbage collection
                    self.tk_image = tk.PhotoImage(data=buffer.tobytes())

                    if self.video_image_id is None:
                        self.video_image_id = self.canvas.create_image(
                            0, 0, anchor="nw", image=self.tk_image, tags="bg"
                        )
                        self.canvas.lower("bg")  # Garantir que fique atrás dos pontos
                    else:
                        self.canvas.itemconfig(self.video_image_id, image=self.tk_image)

        # Agendar próxima atualização (33ms ~ 30 FPS)
        self.window.after(33, self.update_video_feed)

    def start_sequence(self):
        # Limpar pontos anteriores do manager
        self.calib_manager.clear_points()
        self.show_point()

    def show_point(self):
        if self.current_point_idx >= len(self.points):
            self.finish_calibration()
            return

        x, y = self.points[self.current_point_idx]

        # Limpar canvas (exceto textos fixos se houver)
        self.canvas.delete("target")
        self.canvas.delete("countdown")

        # Iniciar animação (Melhoria 14)
        self.animation_start_time = time.time()
        self.animate_point(x, y)

    def animate_point(self, x, y):
        # Duração da animação/countdown antes de coletar
        ANIMATION_DURATION = 1.5 
        
        elapsed = time.time() - self.animation_start_time
        remaining = ANIMATION_DURATION - elapsed
        
        if remaining <= 0:
            # Fim da animação, começar coleta
            self.canvas.delete("target")
            self.canvas.delete("countdown")
            
            # Desenhar ponto fixo final
            r = 20
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill="red", outline="white", tags="target"
            )
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", tags="target")
            
            # Iniciar coleta
            self.frames_collected = 0
            self.update_progress(0)
            self.start_collection()
            return

        # Animação pulsante (15px a 25px)
        pulse = np.sin(elapsed * 10) # Oscilação
        r = 20 + (pulse * 5)
        
        self.canvas.delete("target")
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="red", outline="white", tags="target"
        )
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", tags="target")
        
        # Countdown visual
        self.canvas.delete("countdown")
        count_val = int(remaining) + 1
        self.canvas.create_text(
            x, y, text=str(count_val), fill="white", font=("Arial", 12, "bold"), tags="countdown"
        )
        
        # Próximo frame da animação
        self.window.after(33, lambda: self.animate_point(x, y))

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
            text="Calculando calibração...",
            fill="white",
            font=("Arial", 24),
        )
        self.window.update()
        
        # Chamar callback (o processamento real é feito no main agora para validar)
        self.window.after(100, self.close)

    def close(self):
        self.window.destroy()
        self.on_complete(cancelled=False)

    def on_user_close(self):
        self.window.destroy()
        self.on_complete(cancelled=True)
