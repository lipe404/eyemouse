"""
scroll_controller.py — Controlador de modo dedicado de rolagem vertical (Milestone 5).

Responsabilidades:
  - Modo dedicado: só emite eventos de scroll quando o modo de rolagem estiver explicitamente ativo.
  - Zonas direcionais de tela:
      * Zona Superior (ex: topo 22%): rola para cima.
      * Zona Inferior (ex: base 22%): rola para baixo.
      * Zona Central Neutra (56% central): repouso estável para leitura sem rolagem acidental.
  - Curva de velocidade proporcional/não-linear: velocidade de rolagem aumenta
    à medida que o olhar se aproxima das bordas extremas da tela.
  - Controle de taxa (rate limiting): evita flooding de eventos de roda no Windows SendInput.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ScrollController:
    """
    Controlador de rolagem vertical direcionada por olhar.
    """

    def __init__(
        self,
        screen_h: int = 1080,
        top_zone_ratio: float = 0.22,
        bottom_zone_ratio: float = 0.78,
        min_tick_interval_sec: float = 0.120,
    ):
        self.screen_h = int(screen_h)
        self.top_zone_ratio = float(top_zone_ratio)
        self.bottom_zone_ratio = float(bottom_zone_ratio)
        self.min_tick_interval_sec = float(min_tick_interval_sec)

        self._is_active: bool = False
        self._last_scroll_time: float = 0.0

    @property
    def is_active(self) -> bool:
        return self._is_active

    def activate(self) -> None:
        self._is_active = True
        self._last_scroll_time = 0.0
        logger.info("ScrollController: modo de rolagem ATIVADO.")

    def deactivate(self) -> None:
        self._is_active = False
        logger.info("ScrollController: modo de rolagem DESATIVADO.")

    def toggle(self) -> bool:
        if self._is_active:
            self.deactivate()
        else:
            self.activate()
        return self._is_active

    def update(
        self, y: float, timestamp: Optional[float] = None
    ) -> Optional[int]:
        """
        Avalia a posição vertical do olhar e retorna o delta de scroll se devido.

        Args:
            y: Posição Y atual do cursor na tela.
            timestamp: Timestamp monotônico em segundos.

        Returns:
            Optional[int]: Delta de scroll (positivo=cima, negativo=baixo, None=sem rolagem).
        """
        if not self._is_active:
            return None

        now = time.perf_counter() if timestamp is None else float(timestamp)

        # Respeita o intervalo mínimo entre pulsos de scroll
        if (now - self._last_scroll_time) < self.min_tick_interval_sec:
            return None

        top_boundary = self.screen_h * self.top_zone_ratio
        bottom_boundary = self.screen_h * self.bottom_zone_ratio

        # 1. Zona Superior: Rolar para Cima
        if y < top_boundary:
            # Penetração na zona de 0.0 (na borda interna) até 1.0 (no topo físico)
            penetration = max(0.0, min(1.0, (top_boundary - y) / max(1.0, top_boundary)))
            # Curva de velocidade: 120 (1 dente) até 480 (4 dentes por tick)
            multiplier = 1 + int(penetration * 3.0)
            delta = 120 * multiplier
            self._last_scroll_time = now
            return delta

        # 2. Zona Inferior: Rolar para Baixo
        elif y > bottom_boundary:
            remaining_h = max(1.0, self.screen_h - bottom_boundary)
            penetration = max(0.0, min(1.0, (y - bottom_boundary) / remaining_h))
            multiplier = 1 + int(penetration * 3.0)
            delta = -120 * multiplier
            self._last_scroll_time = now
            return delta

        # 3. Zona Central: Zona morta estável para leitura
        return 0
