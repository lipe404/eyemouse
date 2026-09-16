"""
gesture_engine.py — Motor de gestos, arbitragem de conflitos e estabilização pré-clique (Milestone 5).

Responsabilidades:
  - Arbitragem de Conflitos:
      * Impede que piscadas simultâneas de ambos os olhos gerem dois cliques individuais.
      * Impede que piscadas naturais involuntárias gerem clique duplo acidental.
      * Suprime cliques individuais comuns durante operações de arrasto ativas.
  - Click Freeze (Estabilização de Coordenadas):
      * Durante a oclusão da pálpebra no clique, o rastreamento da íris sofre distorções (fenômeno de Bell).
      * O GestureEngine congela o cursor na posição estável pré-fechamento por ~150ms.
  - Remapeamento e Desativação Individual de Gestos:
      * Permite remapear gestos ("left_blink", "right_blink", "double_blink", "dwell") para ações do sistema.
      * Garante que o sistema nunca exija piscada assimétrica para funcionalidades vitais.
"""

from collections import deque
from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MouseAction(Enum):
    """Ações disparáveis no controlador de mouse."""
    NONE = "NONE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    START_DRAG = "START_DRAG"
    STOP_DRAG = "STOP_DRAG"
    TOGGLE_PRECISION = "TOGGLE_PRECISION"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    PAUSE = "PAUSE"


class GestureEngine:
    """
    Motor central de gestos e arbitragem do EyeMouse.
    """

    def __init__(
        self,
        click_freeze_sec: float = 0.150,
        bilateral_window_sec: float = 0.080,
        enable_double_blink: bool = False,
    ):
        self.click_freeze_sec = float(click_freeze_sec)
        self.bilateral_window_sec = float(bilateral_window_sec)
        self.enable_double_blink = bool(enable_double_blink)

        # Buffer circular de posições estáveis (timestamp, x, y)
        self._pos_buffer = deque(maxlen=15)
        self._freeze_until_time: float = 0.0
        self._frozen_pos: Optional[Tuple[float, float]] = None

        # Tabela de remapeamento de gestos -> Ações
        self._gesture_map: Dict[str, MouseAction] = {
            "left_blink": MouseAction.LEFT_CLICK,
            "right_blink": MouseAction.RIGHT_CLICK,
            "double_blink": MouseAction.DOUBLE_CLICK,
            "hold_start": MouseAction.START_DRAG,
            "hold_end": MouseAction.STOP_DRAG,
            "dwell": MouseAction.LEFT_CLICK,
        }

        # Habilitação individual de gestos
        self._gesture_enabled: Dict[str, bool] = {
            "left_blink": True,
            "right_blink": True,
            "double_blink": self.enable_double_blink,
            "hold_start": True,
            "hold_end": True,
            "dwell": True,
        }

        # Estado de sincronismo bilateral
        self._last_left_event_time: float = 0.0
        self._last_right_event_time: float = 0.0

        # Rastreamento de arraste
        self.is_dragging: bool = False

    # ------------------------------------------------------------------
    # Configuração e Remapeamento
    # ------------------------------------------------------------------

    def set_gesture_action(self, gesture_name: str, action: MouseAction) -> None:
        """Remapeia o gesto indicado para a ação desejada."""
        self._gesture_map[gesture_name] = action
        logger.info("GestureEngine: gesto '%s' remapeado para %s", gesture_name, action.name)

    def set_gesture_enabled(self, gesture_name: str, enabled: bool) -> None:
        """Habilita ou desabilita um gesto individualmente."""
        self._gesture_enabled[gesture_name] = bool(enabled)
        logger.info("GestureEngine: gesto '%s' habilitado=%s", gesture_name, enabled)

    def is_gesture_enabled(self, gesture_name: str) -> bool:
        return self._gesture_enabled.get(gesture_name, False)

    # ------------------------------------------------------------------
    # Buffer de Coordenadas e Click Freeze
    # ------------------------------------------------------------------

    def record_stable_position(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> None:
        """Registra a posição atual do cursor para uso no Click Freeze."""
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._pos_buffer.append((now, float(x), float(y)))

    def trigger_click_freeze(self, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        Ativa o congelamento de cursor para um clique iminente.
        Retorna a coordenada estável pré-fechamento (de ~120-180ms atrás).
        """
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._freeze_until_time = now + self.click_freeze_sec

        # Busca a coordenada no buffer mais próxima de (now - 0.120s)
        target_time = now - 0.120
        best_pos = None

        if self._pos_buffer:
            # Encontra a posição mais estável pré-oclusão
            for t, px, py in reversed(self._pos_buffer):
                if t <= target_time:
                    best_pos = (px, py)
                    break
            if best_pos is None:
                # Usa a mais antiga disponível no buffer
                best_pos = (self._pos_buffer[0][1], self._pos_buffer[0][2])

        if best_pos:
            self._frozen_pos = best_pos
        return self._frozen_pos if self._frozen_pos else (0.0, 0.0)

    def get_stabilized_position(
        self, current_pos: Tuple[float, float], timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Retorna a posição estabilizada:
        Se estiver dentro da janela de Click Freeze, retorna a posição estável congelada.
        Caso contrário, retorna current_pos normalmente.
        """
        now = time.perf_counter() if timestamp is None else float(timestamp)
        if now < self._freeze_until_time and self._frozen_pos is not None:
            return self._frozen_pos
        return current_pos

    # ------------------------------------------------------------------
    # Processamento e Arbitragem de Gestos
    # ------------------------------------------------------------------

    def process_blink_events(
        self,
        left_blink: bool,
        right_blink: bool,
        double_blink: bool,
        hold_start: bool,
        hold_end: bool,
        timestamp: Optional[float] = None,
    ) -> Optional[MouseAction]:
        """
        Recebe os eventos brutos do BlinkDetector e aplica arbitragem estrita.

        Returns:
            MouseAction: Ação decidida para execução, ou None se suprimida.
        """
        now = time.perf_counter() if timestamp is None else float(timestamp)

        # 1. Eventos de Arraste (Hold)
        if hold_start and self._gesture_enabled.get("hold_start", True):
            self.is_dragging = True
            return self._gesture_map.get("hold_start", MouseAction.START_DRAG)

        if hold_end and self._gesture_enabled.get("hold_end", True):
            self.is_dragging = False
            return self._gesture_map.get("hold_end", MouseAction.STOP_DRAG)

        # 2. Arbitragem de Piscada Bilateral
        # Se ambos os olhos piscaram juntos:
        if left_blink and right_blink:
            if self._gesture_enabled.get("double_blink", False):
                self.trigger_click_freeze(timestamp=now)
                return self._gesture_map.get("double_blink", MouseAction.DOUBLE_CLICK)
            else:
                # Piscada bilateral natural: por padrão, descarta para evitar duplo clique acidental
                logger.debug("GestureEngine: piscada bilateral descartada por segurança.")
                return None

        # 3. Arbitragem por Janela Temporal Estrita (Se os olhos fecharam quase juntos)
        if left_blink:
            self._last_left_event_time = now
            if abs(now - self._last_right_event_time) <= self.bilateral_window_sec:
                # Ocorreu quase simultaneamente ao direito: descarta ambos
                return None

        if right_blink:
            self._last_right_event_time = now
            if abs(now - self._last_left_event_time) <= self.bilateral_window_sec:
                return None

        # 4. Durante arraste ativo, ignora cliques individuais normais
        if self.is_dragging and (left_blink or right_blink):
            logger.debug("GestureEngine: clique individual suprimido durante arraste ativo.")
            return None

        # 5. Despacho de Ações Individuais com Click Freeze
        if left_blink and self._gesture_enabled.get("left_blink", True):
            self.trigger_click_freeze(timestamp=now)
            return self._gesture_map.get("left_blink", MouseAction.LEFT_CLICK)

        if right_blink and self._gesture_enabled.get("right_blink", True):
            self.trigger_click_freeze(timestamp=now)
            return self._gesture_map.get("right_blink", MouseAction.RIGHT_CLICK)

        return None
