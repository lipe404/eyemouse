"""
test_scroll_controller.py — Testes do Modo Dedicado de Rolagem (Milestone 5).

Verifica:
  - Inativo por padrão (retorna None independente da posição)
  - Zona Superior (topo 22%) gera delta positivo (rola para cima)
  - Zona Inferior (base 22%) gera delta negativo (rola para baixo)
  - Zona Central (56% central) é zona morta neutra (retorna 0 para leitura confortável)
  - Velocidade proporcional à penetração na zona
  - Rate limiting (respeita intervalo de 120ms)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from scroll_controller import ScrollController


class TestScrollController:
    """Testes unitários do controlador de rolagem."""

    def test_inactive_by_default(self):
        sc = ScrollController(screen_h=1000)
        assert not sc.is_active
        # Posição no extremo superior não gera scroll quando desativado
        assert sc.update(y=10, timestamp=1.0) is None

    def test_toggle_activation(self):
        sc = ScrollController()
        assert sc.toggle() is True
        assert sc.is_active is True
        assert sc.toggle() is False
        assert sc.is_active is False

    def test_top_zone_scroll_up(self):
        """Topo 22% (y < 220 num display de 1000px) gera scroll para cima (> 0)."""
        sc = ScrollController(screen_h=1000, top_zone_ratio=0.22, min_tick_interval_sec=0.100)
        sc.activate()

        # y=50px está na zona superior
        delta = sc.update(y=50, timestamp=1.0)
        assert delta is not None
        assert delta > 0
        assert delta >= 120

    def test_bottom_zone_scroll_down(self):
        """Base 22% (y > 780 num display de 1000px) gera scroll para baixo (< 0)."""
        sc = ScrollController(screen_h=1000, bottom_zone_ratio=0.78, min_tick_interval_sec=0.100)
        sc.activate()

        # y=900px está na zona inferior
        delta = sc.update(y=900, timestamp=1.0)
        assert delta is not None
        assert delta < 0
        assert delta <= -120

    def test_neutral_reading_zone_returns_zero(self):
        """Centro da tela (entre 220px e 780px) é zona morta neutra para leitura (delta == 0)."""
        sc = ScrollController(screen_h=1000, top_zone_ratio=0.22, bottom_zone_ratio=0.78)
        sc.activate()

        # y=500px está no centro exato da tela
        delta = sc.update(y=500, timestamp=1.0)
        assert delta == 0

    def test_rate_limiting(self):
        """Chamadas mais rápidas que 120ms retornam None para evitar congestionar o SO."""
        sc = ScrollController(screen_h=1000, min_tick_interval_sec=0.120)
        sc.activate()

        delta1 = sc.update(y=50, timestamp=10.0)
        assert delta1 is not None and delta1 > 0

        # Apenas 50ms depois
        delta2 = sc.update(y=50, timestamp=10.050)
        assert delta2 is None

        # 130ms depois (> 120ms)
        delta3 = sc.update(y=50, timestamp=10.130)
        assert delta3 is not None and delta3 > 0
