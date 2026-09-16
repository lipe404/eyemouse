"""
test_gesture_engine.py — Testes unitários do GestureEngine (Milestone 5).

Verifica:
  - Arbitragem de conflito bilateral (|delta_t| <= 80ms)
  - Descarte seguro de piscada bilateral involuntária
  - Click Freeze (congelamento de cursor na coordenada pré-oclusão)
  - Supressão de cliques durante arraste ativo
  - Remapeamento e desativação de gestos
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from gesture_engine import GestureEngine, MouseAction


class TestGestureEngineArbitration:
    """Testes de arbitragem de conflitos entre os olhos."""

    def test_simultaneous_bilateral_blink_discarded_by_default(self):
        """Piscada de ambos os olhos simultânea é descartada quando double_blink desabilitado."""
        engine = GestureEngine(enable_double_blink=False)
        action = engine.process_blink_events(
            left_blink=True,
            right_blink=True,
            double_blink=False,
            hold_start=False,
            hold_end=False,
            timestamp=100.0,
        )
        assert action is None

    def test_simultaneous_bilateral_blink_triggers_double_click_when_enabled(self):
        """Quando double_blink estiver explicitamente habilitado, piscada simultânea dispara DOUBLE_CLICK."""
        engine = GestureEngine(enable_double_blink=True)
        action = engine.process_blink_events(
            left_blink=True,
            right_blink=True,
            double_blink=False,
            hold_start=False,
            hold_end=False,
            timestamp=100.0,
        )
        assert action == MouseAction.DOUBLE_CLICK

    def test_bilateral_temporal_coincidence_window(self):
        """Piscadas nos dois olhos com intervalo menor que 80ms devem ser suprimidas (evita 2 cliques)."""
        engine = GestureEngine(bilateral_window_sec=0.080)
        t0 = 100.0

        # Olho esquerdo pisca em t0
        action1 = engine.process_blink_events(True, False, False, False, False, timestamp=t0)
        assert action1 == MouseAction.LEFT_CLICK

        # Olho direito pisca apenas 40ms depois (dentro da janela de 80ms)
        action2 = engine.process_blink_events(False, True, False, False, False, timestamp=t0 + 0.040)
        assert action2 is None  # Suprimido pela janela de coincidência bilateral!

    def test_independent_blinks_outside_window_both_succeed(self):
        """Piscadas separadas por mais de 80ms (ex: 500ms) executam individualmente."""
        engine = GestureEngine(bilateral_window_sec=0.080)
        t0 = 100.0

        action1 = engine.process_blink_events(True, False, False, False, False, timestamp=t0)
        assert action1 == MouseAction.LEFT_CLICK

        action2 = engine.process_blink_events(False, True, False, False, False, timestamp=t0 + 0.500)
        assert action2 == MouseAction.RIGHT_CLICK


class TestClickFreeze:
    """Testes de estabilização de coordenadas pré-clique (Click Freeze)."""

    def test_click_freeze_locks_to_pre_closure_coordinate(self):
        engine = GestureEngine(click_freeze_sec=0.150)
        t0 = 10.0

        # Alimenta coordenadas estáveis antes do fechamento
        engine.record_stable_position(500.0, 500.0, timestamp=t0 - 0.200)
        engine.record_stable_position(502.0, 498.0, timestamp=t0 - 0.150)
        engine.record_stable_position(501.0, 499.0, timestamp=t0 - 0.120)

        # Durante o fechamento da pálpebra (oclusão/Bell), landmarks pulam para (650, 700)
        engine.record_stable_position(650.0, 700.0, timestamp=t0)

        # Dispara o clique em t0
        action = engine.process_blink_events(True, False, False, False, False, timestamp=t0)
        assert action == MouseAction.LEFT_CLICK

        # Durante a janela de freeze (ex: t0 + 0.050s), get_stabilized_position deve retornar
        # a posição estável de ~120ms atrás (~501, 499), não a deformada (650, 700)!
        pos = engine.get_stabilized_position((680.0, 720.0), timestamp=t0 + 0.050)
        assert pos[0] == pytest.approx(501.0, abs=5.0)
        assert pos[1] == pytest.approx(499.0, abs=5.0)

        # Após expirar a janela de freeze (t0 + 0.200s > 0.150s), retorna a coordenada atual normalmente
        pos_after = engine.get_stabilized_position((520.0, 520.0), timestamp=t0 + 0.200)
        assert pos_after == (520.0, 520.0)


class TestDragSuppressionAndRemapping:
    """Testes de supressão durante arrasto e remapeamento."""

    def test_click_suppressed_during_active_drag(self):
        engine = GestureEngine()
        t0 = 50.0

        # Inicia arraste
        act_start = engine.process_blink_events(False, False, False, True, False, timestamp=t0)
        assert act_start == MouseAction.START_DRAG
        assert engine.is_dragging is True

        # Durante o arraste, piscadas acidentais não devem gerar clique
        act_blink = engine.process_blink_events(True, False, False, False, False, timestamp=t0 + 0.5)
        assert act_blink is None

        # Finaliza o arraste
        act_stop = engine.process_blink_events(False, False, False, False, True, timestamp=t0 + 1.0)
        assert act_stop == MouseAction.STOP_DRAG
        assert engine.is_dragging is False

    def test_gesture_disable_and_remapping(self):
        engine = GestureEngine()

        # Desabilita clique esquerdo por piscada
        engine.set_gesture_enabled("left_blink", False)
        assert engine.is_gesture_enabled("left_blink") is False

        action = engine.process_blink_events(True, False, False, False, False, timestamp=10.0)
        assert action is None

        # Remapeia piscada direita para duplo clique
        engine.set_gesture_action("right_blink", MouseAction.DOUBLE_CLICK)
        action_r = engine.process_blink_events(False, True, False, False, False, timestamp=11.0)
        assert action_r == MouseAction.DOUBLE_CLICK
