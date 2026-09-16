"""
dwell_clicker.py — Mecanismo de clique por permanência (Dwell Click) com proteção de rearmamento (Milestone 5).

Responsabilidades:
  - Detecção Temporal de Fixação:
      * Mede a estabilidade espacial do olhar em uma janela temporal contínua (padrão: 900ms).
      * Raio de dispersão espacial configurável (padrão: 30px).
      * Imune a micro-tremores e ruído do sensor dentro do raio.
  - Indicador de Progresso Circular:
      * Fornece progresso normalizado de 0.0 a 1.0 para desenho de anel visual ao redor do cursor.
  - Cancelamento Suave:
      * Cancela o progresso caso o olhar se afaste significativamente do centro de fixação (> 1.35 * R).
  - Proteção de Rearmamento (Anti-Loop):
      * Após disparar um clique, bloqueia novos cliques até que o olhar saia da região (> 1.5 * R)
        ou expire o tempo de rearmamento, impedindo disparos em loop na mesma coordenada.
"""

from enum import Enum
import math
import time
from typing import Callable, Optional, Tuple


class DwellState(Enum):
    """Estados do ciclo de permanência."""
    IDLE = "IDLE"           # Aguardando fixação estável
    DWELLING = "DWELLING"   # Acumulando tempo sobre a âncora
    TRIGGERED = "TRIGGERED" # Clique disparado neste frame
    REARMING = "REARMING"   # Aguardando o olhar sair da zona para rearmar


class DwellClicker:
    """
    Controlador de clique por permanência do olhar.
    """

    def __init__(
        self,
        dwell_time_sec: float = 0.900,
        dwell_radius_px: float = 30.0,
        rearm_distance_px: float = 45.0,
        rearm_timeout_sec: float = 1.200,
        on_click_callback: Optional[Callable[[Tuple[int, int]], None]] = None,
    ):
        self.dwell_time_sec = float(dwell_time_sec)
        self.dwell_radius_px = float(dwell_radius_px)
        self.rearm_distance_px = float(rearm_distance_px)
        self.rearm_timeout_sec = float(rearm_timeout_sec)
        self.on_click_callback = on_click_callback

        self.enabled: bool = True
        self.state: DwellState = DwellState.IDLE

        # Âncora e Acumulador
        self.anchor_pos: Optional[Tuple[float, float]] = None
        self.accumulated_time: float = 0.0
        self.progress: float = 0.0  # 0.0 a 1.0

        # Controle de tempo e rearmamento
        self.last_update_time: Optional[float] = None
        self.last_click_pos: Optional[Tuple[float, float]] = None
        self.last_click_time: float = 0.0

    # ------------------------------------------------------------------
    # API de Ciclo de Vida
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinicia o estado interno do dwell."""
        self.state = DwellState.IDLE
        self.anchor_pos = None
        self.accumulated_time = 0.0
        self.progress = 0.0
        self.last_update_time = None

    def pause(self) -> None:
        """Pausa temporariamente o dwell click."""
        self.enabled = False
        self.reset()

    def resume(self) -> None:
        """Retoma o dwell click."""
        self.enabled = True
        self.reset()

    # ------------------------------------------------------------------
    # Atualização de Posição
    # ------------------------------------------------------------------

    def update(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
        """
        Processa a coordenada atual do olhar.

        Args:
            x: Posição X na tela.
            y: Posição Y na tela.
            timestamp: Timestamp monotônico opcional (segundos).

        Returns:
            Tuple[bool, float, Optional[Tuple[int, int]]]:
              - triggered (bool): True no frame exato em que o clique ocorreu.
              - progress (float): Progresso atual de 0.0 a 1.0.
              - click_pos (Tuple[int, int] ou None): Coordenada do clique se disparado.
        """
        if not self.enabled:
            return False, 0.0, None

        now = time.perf_counter() if timestamp is None else float(timestamp)

        if self.last_update_time is None:
            self.last_update_time = now
            self.anchor_pos = (x, y)
            return False, 0.0, None

        dt = now - self.last_update_time
        self.last_update_time = now

        # Proteção para intervalos atípicos (ex: após pausa prolongada)
        if dt <= 0.0 or dt > 1.0:
            dt = 1.0 / 30.0

        # --- Estado 1: REARMING (Aguardando afastar o olhar após o clique) ---
        if self.state == DwellState.REARMING:
            if self.last_click_pos is not None:
                cx, cy = self.last_click_pos
                dist_from_click = math.hypot(x - cx, y - cy)

                # Rearma se o olhar se afastou significativamente do clique
                # ou se o timeout de rearmamento expirou
                if (
                    dist_from_click >= self.rearm_distance_px
                    or (now - self.last_click_time) >= self.rearm_timeout_sec
                ):
                    self.state = DwellState.IDLE
                    self.anchor_pos = (x, y)
                    self.accumulated_time = 0.0
                    self.progress = 0.0
                    return False, 0.0, None
                else:
                    return False, 0.0, None
            else:
                self.state = DwellState.IDLE
                return False, 0.0, None

        # --- Estado 2: IDLE ou DWELLING ---
        if self.anchor_pos is None:
            self.anchor_pos = (x, y)
            self.accumulated_time = 0.0
            self.progress = 0.0
            self.state = DwellState.IDLE
            return False, 0.0, None

        ax, ay = self.anchor_pos
        dist_from_anchor = math.hypot(x - ax, y - ay)

        # Se estiver dentro do raio de fixação: acumula tempo
        if dist_from_anchor <= self.dwell_radius_px:
            self.state = DwellState.DWELLING
            self.accumulated_time += dt
            self.progress = min(1.0, self.accumulated_time / max(1e-4, self.dwell_time_sec))

            # Atualização suave do centroide da âncora para acompanhar micro-deriva natural
            self.anchor_pos = (
                ax * 0.92 + x * 0.08,
                ay * 0.92 + y * 0.08,
            )

            # Atingiu o tempo total de permanência!
            if self.progress >= 1.0:
                click_target = (int(round(self.anchor_pos[0])), int(round(self.anchor_pos[1])))
                self.state = DwellState.REARMING
                self.last_click_pos = self.anchor_pos
                self.last_click_time = now
                self.progress = 0.0
                self.accumulated_time = 0.0
                self.anchor_pos = None

                if self.on_click_callback:
                    try:
                        self.on_click_callback(click_target)
                    except Exception:
                        pass

                return True, 1.0, click_target

            return False, self.progress, None

        # Se saiu do raio de fixação: cancela suavemente
        elif dist_from_anchor > (self.dwell_radius_px * 1.35):
            # Reinicia âncora na nova posição
            self.anchor_pos = (x, y)
            self.accumulated_time = 0.0
            self.progress = 0.0
            self.state = DwellState.IDLE
            return False, 0.0, None
        else:
            # Zona de transição (entre R e 1.35*R): congela o progresso sem resetar abruptamente
            return False, self.progress, None
