import tkinter as tk
from tkinter import ttk
from config import EMA_ALPHA


class ControlPanel:
    """
    Painel de controle flutuante do aplicativo.

    Permite ajustar parâmetros em tempo real, pausar/retomar, recalibrar
    e visualizar status (FPS, estado dos olhos).
    """

    def __init__(
        self, root, on_pause_toggle, on_recalibrate, on_quit, update_smoothing_cb, on_blink_calibrate
    ):
        """
        Inicializa o painel de controle.

        Args:
            root (tk.Tk): Janela raiz.
            on_pause_toggle (callable): Callback para pausar/retomar.
            on_recalibrate (callable): Callback para recalibrar tela.
            on_quit (callable): Callback para sair do app.
            update_smoothing_cb (callable): Callback para ajustar suavização.
            on_blink_calibrate (callable): Callback para calibrar piscada.
        """
        self.root = root
        self.on_pause_toggle = on_pause_toggle
        self.on_recalibrate = on_recalibrate
        self.on_quit = on_quit
        self.update_smoothing_cb = update_smoothing_cb
        self.on_blink_calibrate = on_blink_calibrate

        self.window = tk.Toplevel(root)
        self.window.title("EyeMouse Control")
        self.window.geometry("300x320") # Aumentado altura para botão extra
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
        
        self.threshold_label = ttk.Label(status_frame, text="Thresh: 0.20")
        self.threshold_label.pack(anchor="w")

        self.mode_label = ttk.Label(
            status_frame, text="Modo: Ativo", foreground="green"
        )
        self.mode_label.pack(anchor="w")

        # Controles
        controls_frame = ttk.LabelFrame(main_frame, text="Ajustes", padding="5")
        controls_frame.pack(fill="x", pady=5)

        ttk.Label(controls_frame, text="Suavização:").pack(anchor="w")
        self.smooth_scale = ttk.Scale(
            controls_frame, from_=0.01, to=1.0, command=self._on_scale_change
        )
        self.smooth_scale.set(EMA_ALPHA)  # Default
        self.smooth_scale.pack(fill="x")

        # Botões
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        # Linha 1 de botões
        row1 = ttk.Frame(btn_frame)
        row1.pack(fill="x", pady=2)

        self.pause_btn = ttk.Button(
            row1, text="Pausar (Ctrl+Shift+P)", command=self._toggle_pause
        )
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=2)

        ttk.Button(row1, text="Sair", command=self.on_quit).pack(
            side="left", expand=True, fill="x", padx=2
        )
        
        # Linha 2 de botões (Calibração)
        row2 = ttk.Frame(btn_frame)
        row2.pack(fill="x", pady=2)
        
        ttk.Button(row2, text="Calibrar Tela", command=self.on_recalibrate).pack(
            side="left", expand=True, fill="x", padx=2
        )
        
        ttk.Button(row2, text="Calibrar Piscada", command=self.on_blink_calibrate).pack(
            side="left", expand=True, fill="x", padx=2
        )

        self.is_paused = False

        # Intercept close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

    def update_pause_text(self, is_paused):
        if is_paused:
            self.pause_btn.config(text="Retomar (Ctrl+Shift+P)")
        else:
            self.pause_btn.config(text="Pausar (Ctrl+Shift+P)")

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.on_pause_toggle:
            self.on_pause_toggle(self.is_paused)
        self.update_pause_text(self.is_paused)

    def _on_scale_change(self, value):
        if self.update_smoothing_cb:
            self.update_smoothing_cb(float(value))

    def update_status(self, fps, left_ear, right_ear, current_threshold=0.20):
        """
        Atualiza os indicadores de status na UI.

        Args:
            fps (float): Frames por segundo atuais.
            left_ear (float): EAR do olho esquerdo.
            right_ear (float): EAR do olho direito.
            current_threshold (float): Threshold atual de piscada.
        """
        self.fps_label.config(text=f"FPS: {int(fps)}")

        # Hack simples: adicionar asterisco se fechado (usando threshold atual)
        l_status = "*" if left_ear < current_threshold else " "
        r_status = "*" if right_ear < current_threshold else " "

        self.eye_status_label.config(
            text=f"EAR E: {left_ear:.2f}{l_status} | D: {right_ear:.2f}{r_status}"
        )
        
        self.threshold_label.config(text=f"Thresh: {current_threshold:.3f}")
