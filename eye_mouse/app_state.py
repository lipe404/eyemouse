"""
app_state.py — Maquina de estados da aplicacao EyeMouse.

Define estados explicitos e transicoes validas.
Centraliza os efeitos colaterais de cada transicao (ex: release_all).

Estados:
  INITIALIZING  — inicializando modulos e camera
  CALIBRATING   — processo de calibracao ativo
  ACTIVE        — rastreamento ativo, eventos de mouse habilitados
  PAUSED        — usuario pausou; sem eventos de mouse
  TRACKING_LOST — rosto nao detectado por tempo configuravel
  ERROR         — falha irrecuperavel; sem eventos de mouse
  SHUTTING_DOWN — encerramento em progresso

Transicoes validas:
  INITIALIZING  -> CALIBRATING, ERROR, SHUTTING_DOWN
  CALIBRATING   -> ACTIVE, ERROR, SHUTTING_DOWN
  ACTIVE        -> PAUSED, CALIBRATING, TRACKING_LOST, ERROR, SHUTTING_DOWN
  PAUSED        -> ACTIVE, CALIBRATING, SHUTTING_DOWN
  TRACKING_LOST -> ACTIVE, PAUSED, ERROR, SHUTTING_DOWN
  ERROR         -> SHUTTING_DOWN
  SHUTTING_DOWN -> (terminal)
"""
from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Callable, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


class AppState(Enum):
    INITIALIZING  = auto()
    CALIBRATING   = auto()
    ACTIVE        = auto()
    PAUSED        = auto()
    TRACKING_LOST = auto()
    ERROR         = auto()
    SHUTTING_DOWN = auto()


class InvalidTransition(Exception):
    """Levantada quando uma transicao de estado invalida e tentada."""

    def __init__(self, from_state: AppState, to_state: AppState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Transicao invalida: {from_state.name} -> {to_state.name}"
        )


# Estados a partir dos quais eventos de mouse sao permitidos
MOUSE_ALLOWED_STATES: FrozenSet[AppState] = frozenset({AppState.ACTIVE})

# Estados que requerem release_all() ao entrar
RELEASE_ON_ENTER: FrozenSet[AppState] = frozenset({
    AppState.PAUSED,
    AppState.TRACKING_LOST,
    AppState.ERROR,
    AppState.SHUTTING_DOWN,
    AppState.CALIBRATING,
})

# Grafo de transicoes validas
_VALID_TRANSITIONS: Dict[AppState, FrozenSet[AppState]] = {
    AppState.INITIALIZING:  frozenset({AppState.CALIBRATING, AppState.ERROR, AppState.SHUTTING_DOWN}),
    AppState.CALIBRATING:   frozenset({AppState.ACTIVE, AppState.ERROR, AppState.SHUTTING_DOWN}),
    AppState.ACTIVE:        frozenset({AppState.PAUSED, AppState.CALIBRATING, AppState.TRACKING_LOST, AppState.ERROR, AppState.SHUTTING_DOWN}),
    AppState.PAUSED:        frozenset({AppState.ACTIVE, AppState.CALIBRATING, AppState.SHUTTING_DOWN}),
    AppState.TRACKING_LOST: frozenset({AppState.ACTIVE, AppState.PAUSED, AppState.ERROR, AppState.SHUTTING_DOWN}),
    AppState.ERROR:         frozenset({AppState.SHUTTING_DOWN}),
    AppState.SHUTTING_DOWN: frozenset(),
}


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------
class StateMachine:
    """
    Maquina de estados thread-safe para o EyeMouse.

    Uso:
        sm = StateMachine(on_release_all=mouse_controller.release_all)
        sm.transition(AppState.CALIBRATING)
        assert sm.state == AppState.CALIBRATING
        assert not sm.mouse_allowed
    """

    def __init__(
        self,
        initial: AppState = AppState.INITIALIZING,
        on_release_all: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            initial:        Estado inicial da maquina.
            on_release_all: Callback chamado sempre que o estado exigir
                            liberacao dos botoes do mouse.
        """
        self._state = initial
        self._on_release_all = on_release_all
        self._lock = threading.RLock()
        self._listeners: List[Callable[[AppState, AppState], None]] = []
        logger.info("StateMachine iniciada em estado: %s", initial.name)

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def mouse_allowed(self) -> bool:
        """True apenas quando eventos de mouse podem ser gerados."""
        return self._state in MOUSE_ALLOWED_STATES

    # ------------------------------------------------------------------
    # Transicoes
    # ------------------------------------------------------------------

    def transition(self, new_state: AppState) -> None:
        """
        Executa uma transicao de estado.

        Args:
            new_state: Estado de destino.

        Raises:
            InvalidTransition: Se a transicao nao for valida.
        """
        with self._lock:
            current = self._state
            if new_state == current:
                return  # Noop; sem re-entrada desnecessaria

            valid = _VALID_TRANSITIONS.get(current, frozenset())
            if new_state not in valid:
                raise InvalidTransition(current, new_state)

            self._state = new_state
            logger.info(
                "Estado: %s -> %s", current.name, new_state.name
            )

            # Efeito colateral: liberar botoes ao entrar em estados criticos
            if new_state in RELEASE_ON_ENTER and self._on_release_all:
                try:
                    self._on_release_all()
                except Exception as exc:
                    logger.error(
                        "Erro em on_release_all durante transicao: %s", exc
                    )

            # Notificar ouvintes (na mesma thread, sem bloqueio externo)
            for listener in self._listeners:
                try:
                    listener(current, new_state)
                except Exception as exc:
                    logger.error("Erro em listener de estado: %s", exc)

    def try_transition(self, new_state: AppState) -> bool:
        """
        Tenta executar uma transicao sem lancar excecao.

        Returns:
            True se a transicao foi realizada, False se invalida.
        """
        try:
            self.transition(new_state)
            return True
        except InvalidTransition:
            return False

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def add_listener(
        self, listener: Callable[[AppState, AppState], None]
    ) -> None:
        """
        Registra um callback chamado em cada transicao.

        Args:
            listener: Funcao(estado_anterior, estado_novo).
        """
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(
        self, listener: Callable[[AppState, AppState], None]
    ) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Diagnostico
    # ------------------------------------------------------------------

    def valid_transitions(self) -> FrozenSet[AppState]:
        """Retorna o conjunto de estados para os quais e possivel transitar."""
        return _VALID_TRANSITIONS.get(self._state, frozenset())

    def __repr__(self) -> str:
        return f"StateMachine(state={self._state.name})"
