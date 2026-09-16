"""
test_dwell_clicker.py — Testes unitários do DwellClicker (Milestone 5).

Verifica:
  - Fixação espacial dentro do raio acumulando tempo até o disparo
  - Indicador de progresso normalizado 0.0 -> 1.0
  - Cancelamento suave ao sair do raio (> 1.35 * R)
  - Proteção de Rearmamento (anti-loop)
  - Reset e pausa
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from dwell_clicker import DwellClicker, DwellState


class TestDwellClicker:
    """Testes completos do mecanismo de Dwell Click."""

    def test_initial_state(self):
        clicker = DwellClicker(dwell_time_sec=0.900, dwell_radius_px=30.0)
        assert clicker.state == DwellState.IDLE
        assert clicker.progress == 0.0

    def test_fixation_accumulates_progress_and_triggers_click(self):
        clicker = DwellClicker(dwell_time_sec=0.900, dwell_radius_px=30.0)
        t = 100.0

        # Primeiro frame define a âncora
        trig, prog, coord = clicker.update(400.0, 300.0, timestamp=t)
        assert not trig
        assert prog == 0.0

        # Permanece fixo por 450ms (~metade do tempo)
        trig, prog, coord = clicker.update(405.0, 298.0, timestamp=t + 0.450)
        assert not trig
        assert prog == pytest.approx(0.50, abs=0.05)
        assert clicker.state == DwellState.DWELLING

        # Permanece fixo até completar 900ms
        trig, prog, coord = clicker.update(402.0, 301.0, timestamp=t + 0.900)
        assert trig is True
        assert prog == 1.0
        assert coord is not None
        assert coord[0] == pytest.approx(402, abs=5)
        assert coord[1] == pytest.approx(300, abs=5)
        assert clicker.state == DwellState.REARMING

    def test_rearm_protection_prevents_endless_loop_clicks(self):
        """Após disparar um clique, manter o olhar no mesmo local NÃO deve disparar outro clique!"""
        clicker = DwellClicker(
            dwell_time_sec=0.500,
            dwell_radius_px=30.0,
            rearm_distance_px=45.0,
            rearm_timeout_sec=1.5,
        )
        t = 10.0

        # Fixa e dispara em 500ms
        clicker.update(200.0, 200.0, timestamp=t)
        trig, _, _ = clicker.update(200.0, 200.0, timestamp=t + 0.500)
        assert trig is True
        assert clicker.state == DwellState.REARMING

        # Mantém o olhar na MESMA posição por mais 500ms, 1s, etc.
        trig2, prog2, _ = clicker.update(202.0, 198.0, timestamp=t + 1.000)
        assert trig2 is False
        assert prog2 == 0.0
        assert clicker.state == DwellState.REARMING

        # Move o olhar para longe (> 45px)
        trig3, prog3, _ = clicker.update(300.0, 200.0, timestamp=t + 1.100)
        assert trig3 is False
        # Agora rearma e entra em IDLE!
        assert clicker.state == DwellState.IDLE

    def test_cancellation_when_gaze_leaves_radius(self):
        """Se o olhar pular para fora do raio (> 1.35 * R), o progresso reseta para 0.0."""
        clicker = DwellClicker(dwell_time_sec=0.500, dwell_radius_px=30.0)
        t = 0.0

        clicker.update(100.0, 100.0, timestamp=t)
        # 6 frames de 50ms = 300ms (> 50% de 500ms)
        for i in range(1, 7):
            _, prog1, _ = clicker.update(102.0, 102.0, timestamp=t + i * 0.050)
        assert prog1 > 0.5

        # Olhar salta para fora do raio
        trig, prog2, _ = clicker.update(250.0, 100.0, timestamp=t + 0.350)
        assert not trig
        assert prog2 == 0.0
        assert clicker.state == DwellState.IDLE

    def test_pause_and_resume(self):
        clicker = DwellClicker()
        clicker.pause()
        assert not clicker.enabled

        trig, prog, _ = clicker.update(100.0, 100.0, timestamp=1.0)
        assert not trig
        assert prog == 0.0

        clicker.resume()
        assert clicker.enabled
