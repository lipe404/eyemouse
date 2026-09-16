"""
mouse_controller.py — Controlador de alto nível do mouse para EyeMouse.

Usa OsMouse (SendInput nativo com latência submilisegundo) ou PyAutoGuiMouse
para injeção de eventos de cursor e botões no sistema operacional.

Princípios e Responsabilidades:
  - Separação estrita dos estágios de coordenadas:
      1. raw_pos (float): coordenada bruta mapeada do olhar pelo modelo de calibração.
      2. filtered_pos (float): coordenada com suavização (OneEuroFilter / Kalman).
      3. os_sent_pos (int): coordenada inteira arredondada enviada ao driver do SO.
  - Precisão float mantida em todo o fluxo, sem arredondamento prematuro.
  - Modo de Precisão (Precision Mode): atenuação de ganho em torno de uma âncora
    para seleção cirúrgica de microelementos (ex: botões de 16x16, hiperlinks).
  - Alcance completo de tela com SCREEN_MARGIN = 0 (de 0 a screen_w - 1).
  - Rastreamento de estado de arrasto (drag & drop) e feedback sonoro assíncrono.
  - Idempotência de release_all() e suporte a emergency_stop() para proteção do usuário.

Aceleração do Mouse do Windows:
  O EyeMouse utiliza coordenadas absolutas (SendInput com MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK).
  Ao contrário de movimentos relativos (dx, dy) que sofrem interferência da curva balística de
  "Aprimorar precisão do ponteiro" do Windows (causando deriva e descalibração rápida), as
  coordenadas absolutas garantem mapeamento 1:1 direto, linear e imune à aceleração do SO.
"""

import logging
import threading
from typing import Optional, Tuple, Union

from os_mouse import BaseMouseBackend, OsMouse, create_mouse_backend
from utils.smoothing import (
    BaseSmoothingFilter,
    KalmanSmoothingFilter,
    OneEuroFilter,
    PassThroughFilter,
    SmoothingFilter,
    create_smoothing_filter,
)
from config import (
    EMA_ALPHA,
    FT_ONE_EURO_FILTER,
    MOUSE_BACKEND,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_ENABLE_DEADZONE,
    ONE_EURO_MIN_CUTOFF,
    PRECISION_MODE_FACTOR,
    SCREEN_MARGIN,
    SMOOTHING_FILTER_TYPE,
)

try:
    import winsound
except ImportError:
    winsound = None

logger = logging.getLogger(__name__)


class MouseController:
    """
    Controlador de alto nível do mouse.

    Integra backend nativo do SO com filtragem adaptativa de olhar, modo de precisão
    e proteções de segurança.
    """

    def __init__(
        self,
        backend: Optional[Union[BaseMouseBackend, str]] = None,
        filter_type: Optional[str] = None,
    ):
        # 1. Inicialização do Backend
        if backend is None:
            if MOUSE_BACKEND == "SENDINPUT":
                self._os_mouse = OsMouse()
            else:
                self._os_mouse = create_mouse_backend(MOUSE_BACKEND)
        elif isinstance(backend, str):
            if backend.upper() in ("SENDINPUT", "OS_MOUSE", "WIN32", "NATIVE"):
                self._os_mouse = OsMouse()
            else:
                self._os_mouse = create_mouse_backend(backend)
        else:
            self._os_mouse = backend

        self.screen_w, self.screen_h = self._os_mouse.screen_size

        # 2. Inicialização do Filtro de Suavização
        self._filter_type: str = (
            filter_type
            if filter_type is not None
            else (SMOOTHING_FILTER_TYPE if FT_ONE_EURO_FILTER else "KALMAN")
        )
        self._smoothing: BaseSmoothingFilter = self._init_filter(self._filter_type)

        # 3. Estado de Arraste e Modo de Precisão
        self._is_dragging: bool = False
        self._precision_mode: bool = False
        self._precision_factor: float = float(PRECISION_MODE_FACTOR)
        self._precision_anchor: Optional[Tuple[float, float]] = None

        # 4. Estágios de Coordenadas (rastreamento de alta resolução)
        self.last_raw_pos: Optional[Tuple[float, float]] = None
        self.last_filtered_pos: Optional[Tuple[float, float]] = None
        self.last_sent_pos: Optional[Tuple[int, int]] = None

        logger.info(
            "MouseController: tela %dx%d, filtro=%s, backend=%s, precisao=%.2fx",
            self.screen_w,
            self.screen_h,
            self._filter_type,
            type(self._os_mouse).__name__,
            self._precision_factor,
        )

    def _init_filter(self, filter_type: str) -> BaseSmoothingFilter:
        f_type = filter_type.upper().strip()
        if f_type in ("ONE_EURO", "1EURO", "ONEEURO"):
            return OneEuroFilter(
                min_cutoff=ONE_EURO_MIN_CUTOFF,
                beta=ONE_EURO_BETA,
                d_cutoff=ONE_EURO_D_CUTOFF,
                enable_deadzone=ONE_EURO_ENABLE_DEADZONE,
            )
        elif f_type in ("KALMAN", "KF"):
            kf = KalmanSmoothingFilter()
            kf.set_alpha(EMA_ALPHA)
            return kf
        elif f_type in ("NONE", "PASSTHROUGH"):
            return PassThroughFilter()
        else:
            return create_smoothing_filter(filter_type)

    # ------------------------------------------------------------------
    # Movimento e Pipeline de Coordenadas
    # ------------------------------------------------------------------

    def move(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> None:
        """
        Move o cursor para (x, y) aplicando suavização, modo de precisão e clamp de tela.

        Estágios:
          1. raw_pos = (x, y) float
          2. filtered_pos = suavização(x, y) float
          3. precision_pos = atenuação por âncora se modo de precisão estiver ativo
          4. clamped_pos = limitação aos limites da tela com SCREEN_MARGIN
          5. sent_pos = round(clamped_pos) -> int enviado ao driver do SO
        """
        raw_x = float(x)
        raw_y = float(y)
        self.last_raw_pos = (raw_x, raw_y)

        # Estágio 2: Suavização (float)
        smooth_x, smooth_y = self._smoothing.update(
            raw_x, raw_y, timestamp=timestamp
        )
        self.last_filtered_pos = (float(smooth_x), float(smooth_y))

        # Estágio 3: Modo de Precisão (float)
        if self._precision_mode:
            if self._precision_anchor is None:
                self._precision_anchor = (smooth_x, smooth_y)
            anchor_x, anchor_y = self._precision_anchor
            eff_x = anchor_x + (smooth_x - anchor_x) * self._precision_factor
            eff_y = anchor_y + (smooth_y - anchor_y) * self._precision_factor
        else:
            eff_x = smooth_x
            eff_y = smooth_y

        # Estágio 4: Limitação de borda (respeita SCREEN_MARGIN)
        margin = float(SCREEN_MARGIN)
        min_x = margin
        max_x = max(min_x, float(self.screen_w - 1) - margin)
        min_y = margin
        max_y = max(min_y, float(self.screen_h - 1) - margin)

        clamped_x = max(min_x, min(max_x, eff_x))
        clamped_y = max(min_y, min(max_y, eff_y))

        # Estágio 5: Conversão inteira e injeção no SO
        final_x = int(round(clamped_x))
        final_y = int(round(clamped_y))
        self.last_sent_pos = (final_x, final_y)

        self._os_mouse.move(final_x, final_y)

    # ------------------------------------------------------------------
    # Modo de Precisão
    # ------------------------------------------------------------------

    def set_precision_mode(
        self, enabled: bool, factor: Optional[float] = None
    ) -> None:
        """
        Ativa ou desativa o Modo de Precisão para seleção de microalvos.

        Args:
            enabled: True ativa; False desativa.
            factor: Multiplicador de escala de movimento (ex: 0.35 = 35% de ganho).
        """
        self._precision_mode = bool(enabled)
        if factor is not None:
            self._precision_factor = max(0.05, min(1.0, float(factor)))
        if not enabled:
            self._precision_anchor = None
        else:
            # Fixa a âncora na posição filtrada atual se disponível
            if self.last_filtered_pos:
                self._precision_anchor = self.last_filtered_pos

    def toggle_precision_mode(self, factor: Optional[float] = None) -> bool:
        """Alterna o estado do modo de precisão e retorna o novo estado."""
        self.set_precision_mode(not self._precision_mode, factor)
        return self._precision_mode

    @property
    def is_precision_mode(self) -> bool:
        return self._precision_mode

    @property
    def precision_factor(self) -> float:
        return self._precision_factor

    # ------------------------------------------------------------------
    # Seleção de Filtros
    # ------------------------------------------------------------------

    def set_filter_type(self, filter_type: str, **kwargs) -> None:
        """
        Altera o algoritmo de suavização em tempo de execução.

        Args:
            filter_type: "ONE_EURO", "KALMAN" ou "NONE".
        """
        self._filter_type = filter_type.upper().strip()
        self._smoothing = create_smoothing_filter(self._filter_type, **kwargs)
        if self._filter_type in ("KALMAN", "KF"):
            self._smoothing.set_alpha(EMA_ALPHA)
        logger.info("MouseController: filtro alterado para %s", self._filter_type)

    def set_smoothing_alpha(self, alpha: float) -> None:
        """Define o fator de suavização do filtro atual."""
        self._smoothing.set_alpha(alpha)

    def reset_smoothing(self) -> None:
        """Reinicia o filtro de suavização e a âncora de precisão."""
        self._smoothing.reset()
        self._precision_anchor = None

    # ------------------------------------------------------------------
    # Ações de Botão
    # ------------------------------------------------------------------

    def left_click(self) -> None:
        """Clique esquerdo."""
        self._os_mouse.left_click()
        self._play_sound(1000, 50)

    def right_click(self) -> None:
        """Clique direito."""
        self._os_mouse.right_click()
        self._play_sound(500, 50)

    def double_click(self) -> None:
        """Duplo clique esquerdo."""
        self._os_mouse.double_click()
        self._play_sound(1500, 50)

    def click(self, button: str = "left") -> None:
        """Realiza um clique no botão especificado ('left' ou 'right')."""
        if button == "left":
            self.left_click()
        elif button == "right":
            self.right_click()
        else:
            logger.warning("Botão desconhecido: %s", button)

    def button_down(self, button: str = "left") -> None:
        """Pressiona e mantém o botão indicado."""
        self._os_mouse.button_down(button)
        if button == "left" and not self._is_dragging:
            self._is_dragging = True
            self._play_sound(800, 200)

    def button_up(self, button: str = "left") -> None:
        """Solta o botão indicado."""
        self._os_mouse.button_up(button)
        if button == "left" and self._is_dragging:
            self._is_dragging = False
            self._play_sound(600, 100)

    def start_drag(self) -> None:
        """Inicia arraste com botão esquerdo."""
        self.button_down("left")

    def stop_drag(self) -> None:
        """Termina arraste com botão esquerdo."""
        self.button_up("left")

    def scroll(self, delta: int) -> None:
        """Rola a roda do mouse (positivo=cima, negativo=baixo)."""
        self._os_mouse.scroll(delta)

    # ------------------------------------------------------------------
    # Segurança
    # ------------------------------------------------------------------

    def release_all(self) -> None:
        """Libera todos os botões mantidos pressionados (idempotente)."""
        self._os_mouse.release_all()
        self._is_dragging = False

    def emergency_stop(self) -> None:
        """Parada emergencial: bloqueia novos eventos e libera botões."""
        self._os_mouse.emergency_stop()
        self._is_dragging = False
        logger.warning("MouseController: emergency_stop ativado.")

    def reset_emergency_stop(self) -> None:
        """Reativa o envio de eventos após emergency_stop()."""
        self._os_mouse.reset_emergency_stop()

    # ------------------------------------------------------------------
    # Propriedades de Estado
    # ------------------------------------------------------------------

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def button_state(self) -> dict:
        return self._os_mouse.button_state

    @property
    def filter_type(self) -> str:
        return self._filter_type

    # ------------------------------------------------------------------
    # Feedback Sonoro
    # ------------------------------------------------------------------

    def _play_sound(self, frequency: int = 1000, duration: int = 100) -> None:
        """Toca um beep assíncrono em thread daemon sem bloquear o pipeline."""
        if not winsound:
            return

        def _run():
            try:
                winsound.Beep(frequency, duration)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

