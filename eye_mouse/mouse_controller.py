"""
mouse_controller.py — Controlador de alto nivel do mouse.

Usa OsMouse (SendInput nativo, zero PAUSE artificial) em vez de pyautogui
para movimento e eventos de botao.

Responsabilidades:
  - Aplicar suavizacao (SmoothingFilter / Kalman).
  - Limitar o cursor aos limites da tela (sem clamp com margem por padrao).
  - Rastrear estado de arrastamento.
  - Emitir feedback sonoro (winsound, assíncrono).
  - Expor release_all() e emergency_stop() para uso pelo StateMachine.

Nao verifica o AppState — essa responsabilidade e do chamador (main.py).
"""

import logging
import threading
from typing import Optional, Tuple

from os_mouse import OsMouse
from utils.smoothing import SmoothingFilter
from config import SCREEN_MARGIN, EMA_ALPHA

try:
    import winsound
except ImportError:
    winsound = None  # Plataformas sem suporte a audio nativo

logger = logging.getLogger(__name__)


class MouseController:
    """
    Controlador de alto nivel do mouse.

    Integra OsMouse (driver nativo) com suavizacao e feedback sonoro.
    """

    def __init__(self):
        self._os_mouse = OsMouse()
        self.screen_w, self.screen_h = self._os_mouse.screen_size
        self._smoothing = SmoothingFilter()
        self._smoothing.set_alpha(EMA_ALPHA)
        self._is_dragging: bool = False

        logger.info(
            "MouseController: tela %dx%d, alpha=%.2f",
            self.screen_w, self.screen_h, EMA_ALPHA,
        )

    # ------------------------------------------------------------------
    # Movimento
    # ------------------------------------------------------------------

    def move(self, x: float, y: float) -> None:
        """
        Move o cursor para (x, y) aplicando suavizacao e limitando as bordas.

        As coordenadas sao limitadas a [0, screen_w-1] x [0, screen_h-1].
        Nao ha margem artificial por padrao — use SCREEN_MARGIN=0 em config.
        Se SCREEN_MARGIN > 0, e aplicado como zona de seguranca interna.
        """
        smooth_x, smooth_y = self._smoothing.update(x, y)

        # Clamp aos limites reais da tela (com margem configuravel)
        margin = SCREEN_MARGIN
        final_x = int(max(margin, min(self.screen_w - 1 - margin, smooth_x)))
        final_y = int(max(margin, min(self.screen_h - 1 - margin, smooth_y)))

        self._os_mouse.move(final_x, final_y)

    # ------------------------------------------------------------------
    # Acoes de botao
    # ------------------------------------------------------------------

    def left_click(self) -> None:
        """Clique esquerdo (alias para compatibilidade com click('left'))."""
        self._os_mouse.left_click()
        self._play_sound(1000, 50)

    def right_click(self) -> None:
        """Clique direito."""
        self._os_mouse.right_click()
        self._play_sound(500, 50)

    def double_click(self) -> None:
        """Duplo clique."""
        self._os_mouse.double_click()
        self._play_sound(1500, 50)

    def click(self, button: str = "left") -> None:
        """
        Realiza um clique.

        Args:
            button: "left" ou "right".
        """
        if button == "left":
            self.left_click()
        elif button == "right":
            self.right_click()
        else:
            logger.warning("Botao desconhecido: %s", button)

    def button_down(self, button: str = "left") -> None:
        """Pressiona e mantem o botao (inicio de arraste)."""
        self._os_mouse.button_down(button)
        if button == "left" and not self._is_dragging:
            self._is_dragging = True
            self._play_sound(800, 200)

    def button_up(self, button: str = "left") -> None:
        """Solta o botao (fim de arraste)."""
        self._os_mouse.button_up(button)
        if button == "left" and self._is_dragging:
            self._is_dragging = False
            self._play_sound(600, 100)

    # Aliases de compatibilidade com codigo legado (main.py existente)
    def start_drag(self) -> None:
        """Inicia arraste com botao esquerdo."""
        self.button_down("left")

    def stop_drag(self) -> None:
        """Termina arraste com botao esquerdo."""
        self.button_up("left")

    def scroll(self, delta: int) -> None:
        """
        Rola a roda do mouse.

        Args:
            delta: 120 = um dente para cima; -120 = um dente para baixo.
        """
        self._os_mouse.scroll(delta)

    # ------------------------------------------------------------------
    # Seguranca
    # ------------------------------------------------------------------

    def release_all(self) -> None:
        """
        Libera todos os botoes pressionados.

        Idempotente. Chamado automaticamente pelo StateMachine em transicoes
        para PAUSED, TRACKING_LOST, ERROR, SHUTTING_DOWN e CALIBRATING.
        """
        self._os_mouse.release_all()
        self._is_dragging = False

    def emergency_stop(self) -> None:
        """
        Para emergencial: bloqueia todos os eventos futuros e libera botoes.

        Use quando o sistema entrar em estado de erro critico.
        Chame reset_emergency_stop() para retomar.
        """
        self._os_mouse.emergency_stop()
        self._is_dragging = False
        logger.warning("MouseController: emergency_stop ativado.")

    def reset_emergency_stop(self) -> None:
        """Reativa o envio de eventos apos emergency_stop()."""
        self._os_mouse.reset_emergency_stop()

    # ------------------------------------------------------------------
    # Suavizacao
    # ------------------------------------------------------------------

    def set_smoothing_alpha(self, alpha: float) -> None:
        """
        Define o fator de suavizacao.

        Args:
            alpha: 0.0 = maximo suave; 1.0 = sem suavizacao.
        """
        self._smoothing.set_alpha(alpha)

    def reset_smoothing(self) -> None:
        """Reinicia o filtro de suavizacao (util apos perda de rastreamento)."""
        self._smoothing.reset()

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def button_state(self) -> dict:
        """Estado atual dos botoes no nivel do OsMouse."""
        return self._os_mouse.button_state

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _play_sound(self, frequency: int = 1000, duration: int = 100) -> None:
        """Toca um beep em thread daemon para nao bloquear o pipeline."""
        if not winsound:
            return

        def _run():
            try:
                winsound.Beep(frequency, duration)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()
