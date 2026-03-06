import tkinter as tk
from tkinter import ttk
from config import EMA_ALPHA

class ControlPanel:
    def __init__(self, root, on_pause_toggle, on_recalibrate, on_quit, update_smoothing_cb):
        self.root = root
        self.on_pause_toggle = on_pause_toggle
        self.on_recalibrate = on_recalibrate
        self.on_quit = on_quit
        self.update_smoothing_cb = update_smoothing_cb
        
        self.window = tk.Toplevel(root)
        self.window.title("EyeMouse Control")
        self.window.geometry("300x250")
        self.window.attributes("-topmost", True)
        self.window.resizable(False, False)
        
        # Posicionar no canto superior direito
        screen_w = self.window.winfo_screenwidth()
        self.window.geometry(f"+{screen_w - 320}+20")
        
        # Frame principal
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Status
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.pack(fill="x", pady=5)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack(anchor="w")
        
        self.eye_status_label = ttk.Label(status_frame, text="Olhos: Detectando...")
        self.eye_status_label.pack(anchor="w")
        
        self.mode_label = ttk.Label(status_frame, text="Modo: Ativo", foreground="green")
        self.mode_label.pack(anchor="w")
        
        # Controles
        controls_frame = ttk.LabelFrame(main_frame, text="Ajustes", padding="5")
        controls_frame.pack(fill="x", pady=5)
        
        ttk.Label(controls_frame, text="Suavização:").pack(anchor="w")
        self.smooth_scale = ttk.Scale(controls_frame, from_=0.01, to=1.0, command=self._on_scale_change)
        self.smooth_scale.set(EMA_ALPHA) # Default
        self.smooth_scale.pack(fill="x")
        
        # Botões
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        self.pause_btn = ttk.Button(btn_frame, text="Pausar", command=self._toggle_pause)
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=2)
        
        ttk.Button(btn_frame, text="Recalibrar", command=self.on_recalibrate).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btn_frame, text="Sair", command=self.on_quit).pack(side="left", expand=True, fill="x", padx=2)
        
        self.is_paused = False
        
        # Intercept close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.config(text="Retomar" if self.is_paused else "Pausar")
        self.mode_label.config(text="Modo: Pausado" if self.is_paused else "Modo: Ativo", 
                               foreground="red" if self.is_paused else "green")
        if self.on_pause_toggle:
            self.on_pause_toggle(self.is_paused)

    def _on_scale_change(self, value):
        if self.update_smoothing_cb:
            self.update_smoothing_cb(float(value))

    def update_status(self, fps, left_open, right_open):
        self.fps_label.config(text=f"FPS: {int(fps)}")
        
        l_status = "Aberto" if left_open else "Fechado"
        r_status = "Aberto" if right_open else "Fechado"
        self.eye_status_label.config(text=f"E: {l_status} | D: {r_status}")
