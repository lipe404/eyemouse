"""
control_panel.py — Painel de controle redesenhado do EyeMouse (Milestone 6).

Responsabilidades:
  - Interface limpa, organizada e com abas de navegação acessíveis:
      * Aba 1: Status & Visão Geral (Rastreamento, Câmera, FPS, Calibração, Perfil, Arraste, Sensibilidade, Latência).
      * Aba 2: Ajustes Rápidos (Perfil, Suavização, Modo Precisão, Barra de Ações).
      * Aba 3: Avançado (Filtro de Suavização, Tempo de Dwell, Limiares de EAR, Calibração Ocular).
      * Aba 4: Perfis & Privacidade (Salvar, Assistente de Configuração, Tela de Treino, Apagar Dados Locais, Termo Legal).
  - Acessibilidade e Teclado:
      * Navegação completa por teclado (Tab, Enter, Espaço, Setas).
      * Botões de alto contraste com tamanhos generosos.
      * Atalho permanente de pausa (Ctrl+Shift+P).
      * Aviso legal explícito de tecnologia assistiva de código aberto.
"""

import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Optional

from config import EMA_ALPHA

logger = logging.getLogger(__name__)


class ControlPanel:
    """
    Painel de controle flutuante e acessível do EyeMouse.
    """

    def __init__(
        self,
        root: tk.Tk,
        on_pause_toggle: Callable[[bool], None],
        on_recalibrate: Callable[[], None],
        on_quit: Callable[[], None],
        update_smoothing_cb: Callable[[float], None],
        on_blink_calibrate: Callable[[], None],
        on_profile_change: Optional[Callable[[str], None]] = None,
        on_action_bar_toggle: Optional[Callable[[bool], None]] = None,
        on_open_wizard: Optional[Callable[[], None]] = None,
        on_open_training: Optional[Callable[[], None]] = None,
        on_clear_data: Optional[Callable[[], None]] = None,
        on_precision_toggle: Optional[Callable[[bool], None]] = None,
    ):
        self.root = root
        self.on_pause_toggle = on_pause_toggle
        self.on_recalibrate = on_recalibrate
        self.on_quit = on_quit
        self.update_smoothing_cb = update_smoothing_cb
        self.on_blink_calibrate = on_blink_calibrate
        self.on_profile_change = on_profile_change
        self.on_action_bar_toggle = on_action_bar_toggle
        self.on_open_wizard = on_open_wizard
        self.on_open_training = on_open_training
        self.on_clear_data = on_clear_data
        self.on_precision_toggle = on_precision_toggle

        self.is_paused: bool = False
        self.is_dragging: bool = False
        self.is_precision: bool = False
        self.action_bar_visible: bool = True

        # Janela principal do painel
        self.window = tk.Toplevel(root)
        self.window.title("EyeMouse Control")
        self.window.geometry("380x480")
        self.window.attributes("-topmost", True)
        self.window.resizable(False, False)

        # Posicionar no canto superior direito
        screen_w = self.window.winfo_screenwidth()
        self.window.geometry(f"+{max(10, screen_w - 400)}+20")

        # Container principal
        self.main_container = ttk.Frame(self.window, padding="8")
        self.main_container.pack(fill="both", expand=True)

        # Notebook de Abas
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill="both", expand=True, pady=(0, 8))

        # Criação das 4 Abas
        self._build_tab_status()
        self._build_tab_quick_adjust()
        self._build_tab_advanced()
        self._build_tab_profiles_privacy()

        # Botões Fixos de Rodapé
        self._build_footer_buttons()

        # Protocolo de fechamento
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

    # ------------------------------------------------------------------
    # Construção das Abas
    # ------------------------------------------------------------------

    def _build_tab_status(self) -> None:
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Visão Geral")

        # StringVars para métricas observáveis
        self.metric_fps_var = tk.StringVar(master=self.window, value="FPS: 0")
        self.metric_latency_var = tk.StringVar(master=self.window, value="Latência Estimada: ~15 ms")
        self.metric_camera_var = tk.StringVar(master=self.window, value="Dispositivo: Câmera #0 (DirectShow)")
        self.metric_calib_var = tk.StringVar(master=self.window, value="Calibração: Validada (< 80px)")
        self.metric_profile_var = tk.StringVar(master=self.window, value="Perfil Ativo: Híbrido")
        self.metric_drag_var = tk.StringVar(master=self.window, value="Arrasto: Normal")
        self.metric_precision_var = tk.StringVar(master=self.window, value="Modo de Precisão: Padrão (1.0x)")

        # Estado do rastreamento
        st_box = ttk.LabelFrame(tab, text="Estado do Rastreamento", padding="8")
        st_box.pack(fill="x", pady=4)

        self.mode_label = ttk.Label(st_box, text="Modo: Ativo", font=("Segoe UI", 10, "bold"), foreground="green")
        self.mode_label.pack(anchor="w")

        self.fps_label = ttk.Label(st_box, textvariable=self.metric_fps_var)
        self.fps_label.pack(anchor="w")

        self.latency_label = ttk.Label(st_box, textvariable=self.metric_latency_var)
        self.latency_label.pack(anchor="w")

        # Qualidade e Câmera
        cam_box = ttk.LabelFrame(tab, text="Câmera e Calibração", padding="8")
        cam_box.pack(fill="x", pady=4)

        self.camera_label = ttk.Label(cam_box, textvariable=self.metric_camera_var)
        self.camera_label.pack(anchor="w")

        self.calib_quality_label = ttk.Label(cam_box, textvariable=self.metric_calib_var)
        self.calib_quality_label.pack(anchor="w")

        # Olhos e Interação
        inter_box = ttk.LabelFrame(tab, text="Dinâmica de Interação", padding="8")
        inter_box.pack(fill="x", pady=4)

        self.profile_display_label = ttk.Label(inter_box, textvariable=self.metric_profile_var)
        self.profile_display_label.pack(anchor="w")

        self.drag_status_label = ttk.Label(inter_box, textvariable=self.metric_drag_var)
        self.drag_status_label.pack(anchor="w")

        self.precision_status_label = ttk.Label(inter_box, textvariable=self.metric_precision_var)
        self.precision_status_label.pack(anchor="w")

        self.filter_display_label = ttk.Label(inter_box, text="Filtro Ativo: One Euro")
        self.filter_display_label.pack(anchor="w")

        self.eye_status_label = ttk.Label(inter_box, text="Olhos: Detectando...")
        self.eye_status_label.pack(anchor="w")

        self.threshold_label = ttk.Label(inter_box, text="Thresh: 0.20")
        self.threshold_label.pack(anchor="w")

    def _build_tab_quick_adjust(self) -> None:
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Ajustes Rápidos")

        # Perfil de Interação
        p_box = ttk.LabelFrame(tab, text="Perfil de Interação", padding="8")
        p_box.pack(fill="x", pady=4)

        self.profile_var = tk.StringVar(master=self.window, value="Híbrido (Padrão)")
        ttk.Label(p_box, text="Modo de Controle:").pack(anchor="w")
        self.profile_combo = ttk.Combobox(
            p_box,
            values=["Híbrido (Padrão)", "Dwell (Sem Piscar)", "Contínuo (Gestos)"],
            textvariable=self.profile_var,
            state="readonly",
        )
        self.profile_combo.current(0)
        self.profile_combo.pack(fill="x", pady=4)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_combo_profile_selected)

        self.action_bar_var = tk.BooleanVar(master=self.window, value=True)
        self.precision_var = tk.BooleanVar(master=self.window, value=False)

        # Suavização
        sm_box = ttk.LabelFrame(tab, text="Suavização e Estabilidade", padding="8")
        sm_box.pack(fill="x", pady=4)

        ttk.Label(sm_box, text="Ajuste de Suavização:").pack(anchor="w")
        self.smooth_scale = ttk.Scale(sm_box, from_=0.01, to=1.0, command=self._on_scale_change)
        self.smooth_scale.set(EMA_ALPHA)
        self.smooth_scale.pack(fill="x", pady=2)

        # Recursos Auxiliares
        aux_box = ttk.LabelFrame(tab, text="Acessibilidade & Ferramentas", padding="8")
        aux_box.pack(fill="x", pady=4)

        self.btn_toggle_precision = ttk.Button(
            aux_box, text="Alternar Modo Precisão (35% ganho)", command=self._toggle_precision
        )
        self.btn_toggle_precision.pack(fill="x", pady=2)

        self.btn_toggle_action_bar = ttk.Button(
            aux_box, text="Ocultar / Exibir Barra de Ações", command=self._toggle_action_bar
        )
        self.btn_toggle_action_bar.pack(fill="x", pady=2)

        if self.on_open_training:
            ttk.Button(
                aux_box, text="Abrir Tela de Treino e Prática", command=self.on_open_training
            ).pack(fill="x", pady=2)

    def _build_tab_advanced(self) -> None:
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Avançado")

        # Algoritmo de Suavização
        f_box = ttk.LabelFrame(tab, text="Algoritmo de Suavização", padding="8")
        f_box.pack(fill="x", pady=4)

        self.filter_display_label = ttk.Label(f_box, text="Filtro Ativo: One Euro Filter (Adaptativo CHI 2012)")
        self.filter_display_label.pack(anchor="w", pady=2)

        # Parâmetros de Dwell Click
        dw_box = ttk.LabelFrame(tab, text="Permanência (Dwell Click)", padding="8")
        dw_box.pack(fill="x", pady=4)

        ttk.Label(dw_box, text="Tempo de Fixação: 900 ms | Raio: 30 px").pack(anchor="w")
        ttk.Label(dw_box, text="Proteção de Rearmamento: 45 px (Anti-Loop)", foreground="#555").pack(anchor="w")

        # Gestos e Calibração Ocular
        gest_box = ttk.LabelFrame(tab, text="Sensibilidade das Pálpebras", padding="8")
        gest_box.pack(fill="x", pady=4)

        ttk.Label(gest_box, text="Histerese Dupla Ativa (± 0.02 EAR)").pack(anchor="w")
        ttk.Button(gest_box, text="Calibrar Sensibilidade de Piscada", command=self.on_blink_calibrate).pack(
            fill="x", pady=4
        )

    def _build_tab_profiles_privacy(self) -> None:
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Privacidade & Perfis")

        # Assistente
        w_box = ttk.LabelFrame(tab, text="Assistente de Configuração", padding="8")
        w_box.pack(fill="x", pady=4)

        ttk.Label(w_box, text="Reconfigure o sistema passo-a-passo:").pack(anchor="w")
        ttk.Button(w_box, text="Iniciar Assistente de Configuração", command=self._start_wizard).pack(
            fill="x", pady=4
        )

        # Privacidade Local e Exclusão de Dados
        priv_box = ttk.LabelFrame(tab, text="Privacidade dos Dados", padding="8")
        priv_box.pack(fill="x", pady=4)

        priv_text = (
            "✓ Todo o processamento de visão roda 100% localmente na CPU.\n"
            "✓ Nenhuma imagem, vídeo ou landmark facial é transmitido para a nuvem.\n"
            "✓ Logs de diagnóstico contêm apenas métricas de desempenho numéricas."
        )
        ttk.Label(priv_box, text=priv_text, font=("Segoe UI", 8), foreground="#2e7d32", justify="left").pack(
            anchor="w", pady=2
        )

        ttk.Button(priv_box, text="Apagar Meus Dados e Calibrações", command=self._confirm_clear_data).pack(
            fill="x", pady=4
        )

        # Aviso Legal de Tecnologia Assistiva
        disc_label = ttk.Label(
            tab,
            text="Aviso Legal: O EyeMouse é um protótipo assistivo de código aberto, não sendo um dispositivo médico homologado.",
            font=("Segoe UI", 7),
            foreground="#777",
            wraplength=340,
            justify="center",
        )
        disc_label.pack(side="bottom", pady=4)

    def _build_footer_buttons(self) -> None:
        btn_frame = ttk.Frame(self.main_container)
        btn_frame.pack(fill="x", pady=(4, 0))

        row1 = ttk.Frame(btn_frame)
        row1.pack(fill="x", pady=2)

        self.pause_btn = ttk.Button(row1, text="Pausar (Ctrl+Shift+P)", command=self._toggle_pause)
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=2)

        ttk.Button(row1, text="Sair", command=self.on_quit).pack(side="left", expand=True, fill="x", padx=2)

        row2 = ttk.Frame(btn_frame)
        row2.pack(fill="x", pady=2)

        ttk.Button(row2, text="Calibrar Tela", command=self.on_recalibrate).pack(
            side="left", expand=True, fill="x", padx=2
        )

    # ------------------------------------------------------------------
    # Ações e Callbacks
    # ------------------------------------------------------------------

    def update_pause_text(self, is_paused: bool) -> None:
        self.is_paused = bool(is_paused)
        if self.is_paused:
            self.pause_btn.config(text="Retomar (Ctrl+Shift+P)")
            self.mode_label.config(text="Modo: Pausado", foreground="#d97706")
        else:
            self.pause_btn.config(text="Pausar (Ctrl+Shift+P)")
            self.mode_label.config(text="Modo: Ativo", foreground="green")

    def _toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.on_pause_toggle:
            self.on_pause_toggle(self.is_paused)
        self.update_pause_text(self.is_paused)

    def _on_scale_change(self, value: Any) -> None:
        if self.update_smoothing_cb:
            self.update_smoothing_cb(float(value))

    def _toggle_precision(self) -> None:
        self.is_precision = not self.is_precision
        self.precision_var.set(self.is_precision)
        if self.on_precision_toggle:
            self.on_precision_toggle(self.is_precision)
        txt = "Precisão Ativa (35% ganho)" if self.is_precision else "Padrão (1.0x)"
        self.metric_precision_var.set(f"Modo de Precisão: {txt}")
        self.precision_status_label.config(text=f"Modo de Precisão: {txt}")

    def _on_precision_toggle(self) -> None:
        self._toggle_precision()

    def _toggle_action_bar(self) -> None:
        self.action_bar_visible = not self.action_bar_visible
        self.action_bar_var.set(self.action_bar_visible)
        if self.on_action_bar_toggle:
            self.on_action_bar_toggle(self.action_bar_visible)

    def _on_action_bar_toggle(self) -> None:
        self._toggle_action_bar()

    def _on_profile_selected(self, event=None) -> None:
        self._on_combo_profile_selected(event)

    def _on_combo_profile_selected(self, event=None) -> None:
        val = self.profile_var.get()
        if "Dwell" in val:
            p_code = "dwell_only"
        elif "Contínuo" in val:
            p_code = "hands_free_blink"
        else:
            p_code = "hybrid"

        self.metric_profile_var.set(f"Perfil Ativo: {val}")
        if self.on_profile_change:
            self.on_profile_change(p_code)

    def _start_wizard(self) -> None:
        if self.on_open_wizard:
            self.on_open_wizard()

    def _on_open_wizard(self) -> None:
        self._start_wizard()

    def _start_training(self) -> None:
        if self.on_open_training:
            self.on_open_training()

    def _on_open_training(self) -> None:
        self._start_training()

    def _confirm_clear_data(self) -> None:
        confirmed = messagebox.askyesno(
            "Apagar Dados",
            "Deseja realmente apagar todos os perfis e calibrações salvas?\nEsta ação não poderá ser desfeita.",
            parent=self.window,
        )
        if confirmed and self.on_clear_data:
            self.on_clear_data()
            messagebox.showinfo("Sucesso", "Dados e calibrações apagados com sucesso.", parent=self.window)

    def _on_clear_data(self) -> None:
        if self.on_clear_data:
            self.on_clear_data()

    # ------------------------------------------------------------------
    # Atualizações Periódicas de Status
    # ------------------------------------------------------------------

    def update_status(self, fps: float, left_ear: float, right_ear: float, current_threshold: float = 0.20) -> None:
        """Atualiza indicadores dinâmicos na UI."""
        self.fps_label.config(text=f"FPS: {int(fps)}")
        self.metric_fps_var.set(f"FPS: {int(fps)}")
        l_status = "*" if left_ear < current_threshold else " "
        r_status = "*" if right_ear < current_threshold else " "
        self.eye_status_label.config(
            text=f"EAR E: {left_ear:.2f}{l_status} | D: {right_ear:.2f}{r_status}"
        )
        self.threshold_label.config(text=f"Thresh: {current_threshold:.3f}")

    def update_face_status(self, face_detected: bool, app_state_name: str) -> None:
        """Atualiza o badge de detecção facial e estado do app."""
        if self.is_paused:
            self.mode_label.config(text="Modo: Pausado", foreground="#d97706")
        elif not face_detected:
            self.mode_label.config(text="Rosto Não Detectado", foreground="#ef4444")
        elif app_state_name == "CALIBRATING":
            self.mode_label.config(text="Modo: Calibrando...", foreground="#3b82f6")
        else:
            self.mode_label.config(text="Modo: Ativo", foreground="green")

    def update_metrics(
        self,
        fps: Optional[float] = None,
        process_fps: Optional[float] = None,
        latency_ms: float = 15.0,
        holdout_error: float = 0.0,
        is_dragging: bool = False,
        is_precision: bool = False,
        profile_name: str = "HYBRID",
        filter_name: str = "ONE_EURO",
        camera_desc: str = "Câmera #0 (DirectShow)",
        **kwargs: Any,
    ) -> None:
        """Atualiza métricas aprofundadas no painel."""
        if fps is not None:
            fps_str = f"FPS: {int(fps)}"
            self.metric_fps_var.set(fps_str)
            self.fps_label.config(text=fps_str)

        lat_str = f"Latência Estimada: ~{latency_ms:.1f} ms" if isinstance(latency_ms, float) else f"Latência Estimada: ~{int(latency_ms)} ms"
        self.metric_latency_var.set(lat_str)
        self.latency_label.config(text=lat_str)
        self.camera_label.config(text=f"Dispositivo: {camera_desc}")

        if holdout_error > 0.0:
            qual = "EXCELENTE (< 40px)" if holdout_error < 40.0 else ("BOA (40-80px)" if holdout_error < 80.0 else "IMPRECISA (> 80px)")
            cal_str = f"Calibração: {holdout_error:.1f}px ({qual})"
            self.metric_calib_var.set(cal_str)
            self.calib_quality_label.config(text=cal_str)

        drag_str = f"Arrasto: {'Arrastando' if is_dragging else 'Normal'}"
        self.metric_drag_var.set(drag_str)
        self.drag_status_label.config(text=drag_str)

        p_txt = "Precisão Ativa (35% ganho)" if is_precision else "Padrão (1.0x)"
        self.metric_precision_var.set(f"Modo de Precisão: {p_txt}")
        self.precision_status_label.config(text=f"Modo de Precisão: {p_txt}")

        if hasattr(self, "filter_display_label"):
            self.filter_display_label.config(text=f"Filtro Ativo: {filter_name}")
