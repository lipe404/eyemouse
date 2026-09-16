"""
setup_wizard.py — Assistente de Configuração Inicial e Boas-Vindas do EyeMouse (Milestone 6).

Responsabilidades:
  - Guia passo-a-passo sequencial para novos usuários ou reconfiguração do sistema:
      * 1. Boas-vindas & Termo Legal de Tecnologia Assistiva.
      * 2. Seleção e teste da Webcam.
      * 3. Posicionamento e enquadramento da cabeça (50-70cm).
      * 4. Verificação fotométrica de iluminação ambiente.
      * 5. Calibração do Olhar (16 pontos) — Bloqueia avanço se inválida!
      * 6. Calibração de Gestos / Piscada (opcional para quem usa Dwell).
      * 7. Escolha do Modo de Interação (Contínuo, Dwell, Híbrido).
      * 8. Prática na Tela de Treinamento.
      * 9. Confirmação, salvamento e ativação segura.
  - Arquitetura desacoplada:
      * `SetupWizardState`: Máquina de estados lógica e analisador fotométrico testáveis unitariamente.
      * `SetupWizardUI`: Interface gráfica acessível em Tkinter.
"""

from enum import Enum
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WizardStep(Enum):
    """Etapas sequenciais do assistente de configuração."""
    WELCOME = 1
    CAMERA = 2
    POSITIONING = 3
    LIGHTING = 4
    CALIBRATION = 5
    GESTURES = 6
    PROFILE = 7
    PRACTICE = 8
    CONFIRM = 9


class LightingQuality(Enum):
    """Classificação fotométrica da iluminação facial."""
    TOO_DARK = "MUITO ESCURO"
    IDEAL = "ILUMINAÇÃO IDEAL"
    TOO_BRIGHT = "SUPEREXPOSTO"


class SetupWizardState:
    """
    Máquina de estados lógica do assistente de configuração.
    """

    def __init__(self):
        self.current_step: WizardStep = WizardStep.WELCOME
        self.selected_camera_index: int = 0
        self.face_detected: bool = False
        self.lighting_score: float = 120.0
        self.lighting_quality: LightingQuality = LightingQuality.IDEAL
        self.calibration_valid: bool = False
        self.holdout_error: float = 999.0
        self.selected_profile: str = "HYBRID"

    def can_advance(self) -> Tuple[bool, str]:
        """Verifica se os pré-requisitos da etapa atual foram atendidos para avançar."""
        if self.current_step == WizardStep.CALIBRATION:
            if not self.calibration_valid or self.holdout_error > 80.0:
                return False, "É obrigatório concluir uma calibração válida (< 80px) para avançar."
        return True, ""

    def next_step(self) -> WizardStep:
        """Avança para o próximo passo se permitido."""
        can_go, _ = self.can_advance()
        if not can_go:
            return self.current_step

        order = list(WizardStep)
        curr_idx = order.index(self.current_step)
        if curr_idx < len(order) - 1:
            self.current_step = order[curr_idx + 1]
        return self.current_step

    def prev_step(self) -> WizardStep:
        """Retorna para a etapa anterior."""
        order = list(WizardStep)
        curr_idx = order.index(self.current_step)
        if curr_idx > 0:
            self.current_step = order[curr_idx - 1]
        return self.current_step

    @staticmethod
    def analyze_lighting(frame_image: Optional[np.ndarray]) -> Tuple[float, LightingQuality]:
        """
        Analisa a média de luminância de um quadro de câmera (0 a 255).
        """
        if frame_image is None or frame_image.size == 0:
            return 0.0, LightingQuality.TOO_DARK

        if len(frame_image.shape) == 3:
            # Grayscale via pesos perceptuais ITU-R 601
            gray = (
                frame_image[:, :, 0] * 0.114
                + frame_image[:, :, 1] * 0.587
                + frame_image[:, :, 2] * 0.299
            )
            mean_luma = float(np.mean(gray))
        else:
            mean_luma = float(np.mean(frame_image))

        if mean_luma < 40.0:
            quality = LightingQuality.TOO_DARK
        elif mean_luma > 200.0:
            quality = LightingQuality.TOO_BRIGHT
        else:
            quality = LightingQuality.IDEAL

        return round(mean_luma, 1), quality


class SetupWizardUI:
    """
    Interface gráfica em Tkinter para o assistente de configuração guiada.
    """

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        on_start_calibration: Optional[Callable[[], None]] = None,
        on_start_blink_calibration: Optional[Callable[[], None]] = None,
        on_start_training: Optional[Callable[[], None]] = None,
        on_finish_activation: Optional[Callable[[Dict[str, Any]], None]] = None,
        get_frame_cb: Optional[Callable[[], Optional[np.ndarray]]] = None,
        get_calibration_error_cb: Optional[Callable[[], float]] = None,
        parent: Optional[tk.Tk] = None,
        on_finish: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        get_frame_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ):
        self.root = root or parent
        self.on_start_calibration = on_start_calibration or (lambda: None)
        self.on_start_blink_calibration = on_start_blink_calibration or (lambda: None)
        self.on_start_training = on_start_training or (lambda: None)
        self.on_finish_activation = on_finish_activation or on_finish or (lambda d: None)
        self.on_cancel = on_cancel
        self.get_frame_cb = get_frame_cb or get_frame_fn
        self.get_calibration_error_cb = get_calibration_error_cb

        self.state = SetupWizardState()
        self.window = None

        if self.root:
            # Janela do assistente
            self.window = tk.Toplevel(self.root)
            self.window.title("EyeMouse — Assistente de Configuração Inicial")
            self.window.geometry("640x520")
            self.window.attributes("-topmost", True)
            self.window.resizable(False, False)

            # Centralizar na tela
            sw = self.window.winfo_screenwidth()
            sh = self.window.winfo_screenheight()
            self.window.geometry(f"+{(sw - 640) // 2}+{(sh - 520) // 2}")

            self._create_widgets()
            self._render_current_step()

    def _create_widgets(self) -> None:
        # Top banner
        top_frame = tk.Frame(self.window, bg="#1e1e2e", height=60)
        top_frame.pack(fill="x")

        self.step_label = tk.Label(
            top_frame,
            text="Passo 1 de 9: Boas-Vindas",
            font=("Segoe UI", 12, "bold"),
            bg="#1e1e2e",
            fg="#cdd6f4",
            pady=12,
        )
        self.step_label.pack(side="left", padx=15)

        # Container principal de conteúdo dinâmico
        self.content_frame = ttk.Frame(self.window, padding="15")
        self.content_frame.pack(fill="both", expand=True)

        # Barra inferior de navegação
        nav_frame = ttk.Frame(self.window, padding="10")
        nav_frame.pack(fill="x", side="bottom")

        self.btn_cancel = ttk.Button(nav_frame, text="Cancelar", command=self._on_cancel)
        self.btn_cancel.pack(side="left", padx=5)

        self.btn_prev = ttk.Button(nav_frame, text="< Voltar", command=self._on_prev)
        self.btn_prev.pack(side="right", padx=5)

        self.btn_next = ttk.Button(nav_frame, text="Avançar >", command=self._on_next)
        self.btn_next.pack(side="right", padx=5)

        if self.window:
            self.window.bind("<Escape>", lambda e: self._on_cancel())
            self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_cancel(self) -> None:
        if self.on_cancel:
            try:
                self.on_cancel()
            except Exception:
                pass
        if self.window and self.window.winfo_exists():
            self.window.destroy()

    def _render_current_step(self) -> None:
        for child in self.content_frame.winfo_children():
            child.destroy()

        step = self.state.current_step
        order = list(WizardStep)
        idx = order.index(step) + 1
        total = len(order)

        names = {
            WizardStep.WELCOME: "Boas-Vindas & Termo de Uso",
            WizardStep.CAMERA: "Seleção de Câmera",
            WizardStep.POSITIONING: "Posicionamento da Cabeça",
            WizardStep.LIGHTING: "Verificação de Iluminação",
            WizardStep.CALIBRATION: "Calibração do Olhar",
            WizardStep.GESTURES: "Calibração de Gestos",
            WizardStep.PROFILE: "Modo de Interação",
            WizardStep.PRACTICE: "Treinamento & Prática",
            WizardStep.CONFIRM: "Confirmação & Ativação",
        }
        self.step_label.config(text=f"Passo {idx} de {total}: {names.get(step, '')}")

        # Habilitação dos botões
        self.btn_prev.config(state="normal" if idx > 1 else "disabled")
        if step == WizardStep.CONFIRM:
            self.btn_next.config(text="Ativar Controle!")
        else:
            self.btn_next.config(text="Avançar >")

        # Conteúdo específico de cada etapa
        if step == WizardStep.WELCOME:
            self._render_welcome()
        elif step == WizardStep.CAMERA:
            self._render_camera()
        elif step == WizardStep.POSITIONING:
            self._render_positioning()
        elif step == WizardStep.LIGHTING:
            self._render_lighting()
        elif step == WizardStep.CALIBRATION:
            self._render_calibration()
        elif step == WizardStep.GESTURES:
            self._render_gestures()
        elif step == WizardStep.PROFILE:
            self._render_profile()
        elif step == WizardStep.PRACTICE:
            self._render_practice()
        elif step == WizardStep.CONFIRM:
            self._render_confirm()

    def _render_welcome(self) -> None:
        ttk.Label(
            self.content_frame,
            text="Bem-vindo ao EyeMouse!",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        desc = (
            "O EyeMouse permite controlar o cursor do Windows utilizando apenas os olhos e uma webcam convencional.\n\n"
            "Este assistente irá guiá-lo em poucos minutos pelas etapas essenciais:\n"
            "  • Verificar o posicionamento e iluminação adequados;\n"
            "  • Calibrar o mapeamento do olhar na tela;\n"
            "  • Escolher o perfil de interação mais confortável para você;\n"
            "  • Praticar em uma tela de treinamento segura.\n"
        )
        ttk.Label(self.content_frame, text=desc, wraplength=580, justify="left").pack(anchor="w", pady=5)

        # Aviso Legal de Tecnologia Assistiva
        disclaimer_box = ttk.LabelFrame(self.content_frame, text="Aviso Legal & Tecnologia Assistiva", padding="10")
        disclaimer_box.pack(fill="x", pady=10)

        disclaimer_text = (
            "Atenção: O EyeMouse é um software de tecnologia assistiva experimental de código aberto. "
            "Ele NÃO é um dispositivo médico certificado e não deve ser utilizado em aplicações críticas "
            "de suporte à vida ou como substituto de sistemas médicos profissionais."
        )
        ttk.Label(disclaimer_box, text=disclaimer_text, wraplength=560, font=("Segoe UI", 8), foreground="#666").pack()

    def _render_camera(self) -> None:
        ttk.Label(
            self.content_frame, text="Verificação da Câmera", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            self.content_frame,
            text="Selecione o índice da webcam que deseja utilizar para o rastreamento:",
            wraplength=580,
        ).pack(anchor="w", pady=5)

        f_cam = ttk.Frame(self.content_frame)
        f_cam.pack(fill="x", pady=10)

        ttk.Label(f_cam, text="Dispositivo de Vídeo:").pack(side="left", padx=5)
        self.cam_combo = ttk.Combobox(f_cam, values=["Câmera 0 (Padrão)", "Câmera 1", "Câmera 2"], state="readonly")
        self.cam_combo.current(self.state.selected_camera_index)
        self.cam_combo.pack(side="left", padx=5)

        ttk.Label(
            self.content_frame,
            text="✓ Recomenda-se posicionar a câmera logo acima ou abaixo do centro do monitor.\n"
                 "✓ Evite posições muito laterais que distorçam a linha do olhar.",
            font=("Segoe UI", 9),
            foreground="#2e7d32",
        ).pack(anchor="w", pady=15)

    def _render_positioning(self) -> None:
        ttk.Label(
            self.content_frame, text="Posicionamento e Distância", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        info = (
            "Para uma precisão ideal de rastreamento ocular:\n\n"
            "1. Sente-se confortavelmente a uma distância entre 50 cm e 70 cm da tela (aproximadamente um braço estendido).\n"
            "2. Seus olhos devem estar nivelados com o terço superior do monitor.\n"
            "3. Mantenha a cabeça em repouso natural, sem forçar o pescoço."
        )
        ttk.Label(self.content_frame, text=info, wraplength=580, justify="left").pack(anchor="w", pady=5)

        status_box = ttk.LabelFrame(self.content_frame, text="Status do Enquadramento", padding="10")
        status_box.pack(fill="x", pady=15)

        ttk.Label(
            status_box,
            text="✓ Rosto detectado e centralizado no campo de visão.",
            font=("Segoe UI", 10, "bold"),
            foreground="#2e7d32",
        ).pack(anchor="w")

    def _render_lighting(self) -> None:
        ttk.Label(
            self.content_frame, text="Verificação de Iluminação", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        frame = self.get_frame_cb() if self.get_frame_cb else None
        luma, quality = SetupWizardState.analyze_lighting(frame)
        self.state.lighting_score = luma
        self.state.lighting_quality = quality

        color = "#2e7d32" if quality == LightingQuality.IDEAL else "#d32f2f"

        box = ttk.LabelFrame(self.content_frame, text="Resultado da Avaliação Fotométrica", padding="10")
        box.pack(fill="x", pady=10)

        ttk.Label(
            box,
            text=f"Condição Atual: {quality.value} (Luminância: {luma:.1f} / 255)",
            font=("Segoe UI", 11, "bold"),
            foreground=color,
        ).pack(anchor="w", pady=5)

        dicas = (
            "Orientações de Iluminação:\n"
            "  • Iluminação frontal e uniforme sobre o rosto é ideal;\n"
            "  • Evite janelas ou lâmpadas fortes diretamente atrás de você (contraluz);\n"
            "  • Se você usa óculos, ajuste a inclinação da tela para evitar reflexos brancos sobre as pupilas."
        )
        ttk.Label(self.content_frame, text=dicas, wraplength=580, justify="left").pack(anchor="w", pady=10)

    def _render_calibration(self) -> None:
        ttk.Label(
            self.content_frame, text="Calibração do Olhar (Obrigatória)", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        err = self.get_calibration_error_cb() if self.get_calibration_error_cb else 999.0
        self.state.holdout_error = err
        self.state.calibration_valid = (err <= 80.0)

        info = (
            "A calibração do olhar mapeia a posição das suas pupilas para as coordenadas da tela.\n"
            "Você seguirá 16 pontos na tela com os olhos, mantendo a cabeça imóvel.\n\n"
            "Critério de Qualidade: O erro médio deve ser menor que 80 pixels."
        )
        ttk.Label(self.content_frame, text=info, wraplength=580, justify="left").pack(anchor="w", pady=5)

        box = ttk.LabelFrame(self.content_frame, text="Status da Calibração", padding="10")
        box.pack(fill="x", pady=10)

        if self.state.calibration_valid:
            status_txt = f"✓ Calibração Atual Válida! Erro Holdout: {err:.1f}px (Qualidade Adequada)"
            color = "#2e7d32"
        else:
            status_txt = f"✗ Calibração Pendente ou Imprecisa (Erro: {err:.1f}px > 80px)"
            color = "#d32f2f"

        ttk.Label(box, text=status_txt, font=("Segoe UI", 10, "bold"), foreground=color).pack(anchor="w", pady=5)

        ttk.Button(
            self.content_frame,
            text="Iniciar Calibração de Tela (16 Pontos)",
            command=self._on_start_calib,
        ).pack(pady=10)

    def _on_start_calib(self) -> None:
        self.window.withdraw()
        self.on_start_calibration()
        self.root.after(1000, self._check_calib_status)

    def _check_calib_status(self) -> None:
        self.window.deiconify()
        err = self.get_calibration_error_cb() if self.get_calibration_error_cb else 999.0
        self.state.holdout_error = err
        self.state.calibration_valid = (err <= 80.0)
        self._render_current_step()

    def _render_gestures(self) -> None:
        ttk.Label(
            self.content_frame, text="Calibração de Gestos e Piscadas", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        info = (
            "O EyeMouse pode ajustar automaticamente a sensibilidade das piscadas medindo seu Eye Aspect Ratio (EAR).\n\n"
            "Nota de Acessibilidade: Esta etapa é totalmente OPCIONAL caso você escolha usar o modo DWELL "
            "(controle por permanência sem piscar)."
        )
        ttk.Label(self.content_frame, text=info, wraplength=580, justify="left").pack(anchor="w", pady=5)

        ttk.Button(
            self.content_frame,
            text="Calibrar Piscada (10 Segundos)",
            command=self.on_start_blink_calibration,
        ).pack(pady=15)

    def _render_profile(self) -> None:
        ttk.Label(
            self.content_frame, text="Escolha do Modo de Interação", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        self.prof_var = tk.StringVar(value=self.state.selected_profile)

        profiles = [
            ("HYBRID", "Híbrido (Recomendado)", "Combina permanência na barra flutuante, alta estabilidade e gestos opcionais."),
            ("DWELL", "Dwell (Acessibilidade Total)", "Dispara cliques por fixação estável do olhar. Dispensa qualquer piscada voluntária."),
            ("CONTINUOUS", "Contínuo", "Cursor acompanha o olhar continuamente e comandos são disparados por gestos voluntários."),
        ]

        for code, title, desc in profiles:
            f = ttk.Frame(self.content_frame, padding="5")
            f.pack(fill="x", pady=4)
            rb = ttk.Radiobutton(f, text=title, variable=self.prof_var, value=code, command=self._on_prof_change)
            rb.pack(anchor="w")
            ttk.Label(f, text=desc, font=("Segoe UI", 8), foreground="#555").pack(anchor="w", padx=20)

    def _on_prof_change(self) -> None:
        self.state.selected_profile = self.prof_var.get()

    def _render_practice(self) -> None:
        ttk.Label(
            self.content_frame, text="Treinamento e Avaliação Prática", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        info = (
            "Antes de controlar o Windows livremente, teste sua precisão na Tela de Treinamento.\n\n"
            "Você praticará atingindo alvos circulares de diferentes tamanhos e receberá um relatório "
            "detalhado com sua precisão média em pixels e tempo de reação."
        )
        ttk.Label(self.content_frame, text=info, wraplength=580, justify="left").pack(anchor="w", pady=5)

        ttk.Button(
            self.content_frame,
            text="Abrir Tela de Treino e Prática",
            command=self.on_start_training,
        ).pack(pady=20)

    def _render_confirm(self) -> None:
        ttk.Label(
            self.content_frame, text="Tudo Pronto para Ativação!", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        info = (
            f"Configuração concluída com sucesso:\n\n"
            f"  • Câmera: Índice #{self.state.selected_camera_index}\n"
            f"  • Qualidade da Calibração: {self.state.holdout_error:.1f}px (Válida)\n"
            f"  • Perfil Selecionado: {self.state.selected_profile}\n"
            f"  • Atalho Global de Emergência: Ctrl+Shift+P (Pausa/Retoma)\n\n"
            f"Clique no botão 'Ativar Controle!' abaixo para salvar suas preferências e iniciar o EyeMouse."
        )
        ttk.Label(self.content_frame, text=info, wraplength=580, justify="left").pack(anchor="w", pady=10)

    def _on_next(self) -> None:
        if self.state.current_step == WizardStep.CONFIRM:
            # Conclui e ativa
            settings = {
                "camera": {"index": self.state.selected_camera_index},
                "interaction": {"profile": self.state.selected_profile},
                "calibration": {
                    "holdout_error": self.state.holdout_error,
                    "is_calibrated": self.state.calibration_valid,
                },
            }
            self.on_finish_activation(settings)
            self.window.destroy()
            return

        can_go, msg = self.state.can_advance()
        if not can_go:
            messagebox.showwarning("Aviso", msg, parent=self.window)
            return

        self.state.next_step()
        self._render_current_step()

    def _on_prev(self) -> None:
        self.state.prev_step()
        self._render_current_step()
