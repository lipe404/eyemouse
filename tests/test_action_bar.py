"""
test_action_bar.py — Testes da Barra de Ações (Milestone 5).

Verifica:
  - Modelo Next-Action: armar ação e auto-reversão para LEFT_CLICK após consumo
  - Ancoragem cíclica: TOP -> BOTTOM -> LEFT -> RIGHT -> TOP
  - Sincronização de estados de arraste, rolagem, precisão e pausa
  - Hit-test e Dwell Click handling
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from gesture_engine import MouseAction
from ui.action_bar import ActionBar, DockPosition


class TestActionBar:
    """Testes unitários da ActionBar."""

    def test_initial_state_defaults_to_left_click(self):
        bar = ActionBar(root=None)
        assert bar.armed_action == MouseAction.LEFT_CLICK
        assert bar.dock_position == DockPosition.TOP
        assert bar.is_dragging is False
        assert bar.is_scrolling is False

    def test_next_action_arming_and_auto_revert(self):
        """Ao armar clique direito, o consumo retorna RIGHT_CLICK e reverte para LEFT_CLICK."""
        bar = ActionBar(root=None)
        bar.arm_action(MouseAction.RIGHT_CLICK)
        assert bar.armed_action == MouseAction.RIGHT_CLICK

        # Primeiro consumo executa a ação armada
        consumed = bar.consume_armed_action()
        assert consumed == MouseAction.RIGHT_CLICK

        # Reverteu automaticamente para LEFT_CLICK!
        assert bar.armed_action == MouseAction.LEFT_CLICK
        assert bar.consume_armed_action() == MouseAction.LEFT_CLICK

    def test_double_click_arming_and_auto_revert(self):
        bar = ActionBar(root=None)
        bar.arm_action(MouseAction.DOUBLE_CLICK)
        assert bar.armed_action == MouseAction.DOUBLE_CLICK

        consumed = bar.consume_armed_action()
        assert consumed == MouseAction.DOUBLE_CLICK
        assert bar.armed_action == MouseAction.LEFT_CLICK

    def test_cycle_dock_positions(self):
        """Alterna ciclicamente entre as 4 posições de ancoragem da tela."""
        bar = ActionBar(root=None)
        assert bar.dock_position == DockPosition.TOP

        assert bar.cycle_dock() == DockPosition.BOTTOM
        assert bar.cycle_dock() == DockPosition.LEFT
        assert bar.cycle_dock() == DockPosition.RIGHT
        assert bar.cycle_dock() == DockPosition.TOP

    def test_state_toggles(self):
        mock_drag = MagicMock()
        mock_scroll = MagicMock()
        mock_prec = MagicMock()
        mock_pause = MagicMock()

        bar = ActionBar(
            root=None,
            on_drag_toggle=mock_drag,
            on_scroll_toggle=mock_scroll,
            on_precision_toggle=mock_prec,
            on_pause_toggle=mock_pause,
        )

        bar.set_dragging(True)
        assert bar.is_dragging is True

        bar.set_scrolling(True)
        assert bar.is_scrolling is True

        bar.set_precision(True)
        assert bar.is_precision is True

        bar.set_paused(True)
        assert bar.is_paused is True

    def test_dwell_hit_test_and_handling(self):
        """Valida que quando um dwell click atinge a barra, aciona o botão e consome o clique."""
        bar = ActionBar(root=None)
        # Mock do cache de bounds dos botões
        bar._btn_bounds = {
            "RIGHT_CLICK": (100, 10, 180, 55),
            "DRAG": (185, 10, 260, 55),
        }
        # Força visibilidade lógica
        bar.is_visible = lambda: True

        # Dwell fora da barra
        hit_outside = bar.handle_dwell_click(500, 500)
        assert hit_outside is False
        assert bar.armed_action == MouseAction.LEFT_CLICK

        # Dwell dentro do botão RIGHT_CLICK
        hit_right = bar.handle_dwell_click(140, 30)
        assert hit_right is True
        assert bar.armed_action == MouseAction.RIGHT_CLICK
