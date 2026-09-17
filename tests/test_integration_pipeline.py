"""
test_integration_pipeline.py — Testes de integração ponta a ponta do EyeMouse.

Milestone 7:
  - Injeção de dependências (MockMouseDriver) sem envio de eventos reais ao Windows.
  - Relógio determinístico simulado (SimulatedClock) sem dependência de time.sleep().
  - Validação de resultados numéricos, transições de estado e proteção contra eventos presos.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

# Adiciona o diretório do código fonte
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eye_mouse")))

from app_state import AppState, StateMachine
from calibration import CalibrationManager
from calibration_models import PolynomialCalibrationModel
from dwell_clicker import DwellClicker
from frame_data import FramePacket, TrackingResult
from gesture_engine import GestureEngine, MouseAction
from mouse_controller import MouseController
from os_mouse import BaseMouseBackend
from tracking_validator import TrackingValidator
from utils.smoothing import OneEuroFilter


# ---------------------------------------------------------------------------
# Driver de Mouse Simulado (Injeção de Dependência)
# ---------------------------------------------------------------------------
class MockMouseDriver(BaseMouseBackend):
    """Driver de teste que captura todas as operações sem chamar a Win32 API."""

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._left_down = False
        self._right_down = False
        self._emergency_stopped = False

        # Logs de eventos para asserções
        self.moves: List[Tuple[int, int]] = []
        self.left_clicks = 0
        self.right_clicks = 0
        self.double_clicks = 0
        self.scrolls: List[int] = []
        self.button_downs: List[str] = []
        self.button_ups: List[str] = []
        self.release_all_calls = 0

    def move(self, x: int, y: int) -> None:
        if not self._emergency_stopped:
            self.moves.append((int(x), int(y)))

    def left_click(self) -> None:
        if not self._emergency_stopped:
            self.left_clicks += 1

    def right_click(self) -> None:
        if not self._emergency_stopped:
            self.right_clicks += 1

    def double_click(self) -> None:
        if not self._emergency_stopped:
            self.double_clicks += 1

    def button_down(self, button: str = "left") -> None:
        if not self._emergency_stopped:
            self.button_downs.append(button)
            if button == "left":
                self._left_down = True
            elif button == "right":
                self._right_down = True

    def button_up(self, button: str = "left") -> None:
        self.button_ups.append(button)
        if button == "left":
            self._left_down = False
        elif button == "right":
            self._right_down = False

    def scroll(self, delta: int) -> None:
        if not self._emergency_stopped:
            self.scrolls.append(delta)

    def release_all(self) -> None:
        self.release_all_calls += 1
        if self._left_down:
            self.button_ups.append("left")
            self._left_down = False
        if self._right_down:
            self.button_ups.append("right")
            self._right_down = False

    def emergency_stop(self) -> None:
        self._emergency_stopped = True
        self.release_all()

    def reset_emergency_stop(self) -> None:
        self._emergency_stopped = False

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    @property
    def button_state(self) -> dict:
        return {"left_down": self._left_down, "right_down": self._right_down}

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._screen_w, self._screen_h

    @property
    def virtual_screen_geometry(self) -> Tuple[int, int, int, int]:
        return 0, 0, self._screen_w, self._screen_h


# ---------------------------------------------------------------------------
# Relógio Controlável para Testes Determinísticos
# ---------------------------------------------------------------------------
class SimulatedClock:
    """Relógio determinístico que avança sob controle do teste."""

    def __init__(self, start_time: float = 1000.0):
        self.current_time = float(start_time)

    def now(self) -> float:
        return self.current_time

    def advance(self, delta_sec: float) -> float:
        self.current_time += float(delta_sec)
        return self.current_time


# ---------------------------------------------------------------------------
# Testes de Integração
# ---------------------------------------------------------------------------
class TestIntegrationPipeline:

    def test_full_pipeline_gaze_to_cursor_movement(self):
        """
        Testa o fluxo completo: Features -> Calibração -> One Euro Filter -> MouseController -> MockDriver.
        Verifica posições numéricas contínuas sem erros e sem ultrapassar a tela.
        """
        driver = MockMouseDriver(screen_w=1920, screen_h=1080)
        mouse = MouseController(backend=driver, filter_type="ONE_EURO")

        # Configura modelo calibrado conhecido
        calib = CalibrationManager(profile_name="integration_test")
        # Simula grade linear conhecida: x_tela = 1920 * iris_x, y_tela = 1080 * iris_y
        calib.model = PolynomialCalibrationModel()
        # [c0, c1, c2, c3, c4, c5] -> c1=1920 (termo x), c2=0 (termo y)
        calib.model.coeffs_x = np.array([0.0, 1920.0, 0.0, 0.0, 0.0, 0.0])
        calib.model.coeffs_y = np.array([0.0, 0.0, 1080.0, 0.0, 0.0, 0.0])
        calib.model.is_fitted = True
        calib.is_calibrated = True

        clock = SimulatedClock(100.0)

        # Simula 30 frames de movimento contínuo da íris de (0.2, 0.2) até (0.8, 0.8)
        for i in range(30):
            t = clock.advance(0.033)
            alpha = i / 29.0
            iris_x = 0.2 + 0.6 * alpha
            iris_y = 0.2 + 0.6 * alpha

            # Mapeamento
            screen_pos = calib.map_to_screen((iris_x, iris_y))
            assert screen_pos is not None

            # Controle do mouse
            mouse.move(screen_pos[0], screen_pos[1], timestamp=t)

        # Validações numéricas
        assert len(driver.moves) == 30
        first_pos = driver.moves[0]
        last_pos = driver.moves[-1]

        # Posições devem estar dentro da tela 1920x1080
        for x, y in driver.moves:
            assert 0 <= x < 1920
            assert 0 <= y < 1080

        # Movimento deve ter progredido em direção ao quadrante inferior direito
        assert last_pos[0] > first_pos[0]
        assert last_pos[1] > first_pos[1]

    def test_dwell_click_lifecycle_integration(self):
        """
        Testa o ciclo de vida do clique por permanência (Dwell Click):
        1. Fixação contínua por 900 ms dispara clique esquerdo.
        2. Fixação contínua subsequente dentro do mesmo raio NÃO dispara novo clique (anti-loop).
        3. Afastamento > 45 px rearma o Dwell.
        4. Nova fixação de 900 ms dispara o segundo clique.
        """
        driver = MockMouseDriver()
        dwell = DwellClicker(
            dwell_time_sec=0.900,
            dwell_radius_px=30.0,
            rearm_distance_px=45.0,
            rearm_timeout_sec=1.2,
        )
        clock = SimulatedClock(50.0)

        # 1. Fixação de 800 ms (ainda não atingiu 900 ms)
        for i in range(24):  # 24 * 33.3ms ≈ 800ms
            t = clock.advance(0.0333)
            triggered, pct, _ = dwell.update(500.0, 500.0, timestamp=t)
            assert not triggered
            assert 0.0 <= pct < 1.0
        assert pct > 0.7  # Ao final dos 800ms, o progresso deve ser alto

        # Avança mais frames para ultrapassar 900 ms
        click_fired = False
        for _ in range(8):
            t = clock.advance(0.0333)
            triggered, pct, _ = dwell.update(500.0, 500.0, timestamp=t)
            if triggered:
                click_fired = True
                driver.left_click()

        assert click_fired is True
        assert driver.left_clicks == 1

        # 2. Continua fixado na mesma posição (anti-loop deve impedir segundo clique)
        for _ in range(30):  # Mais 1 segundo parado no mesmo ponto
            t = clock.advance(0.0333)
            triggered, pct, _ = dwell.update(502.0, 498.0, timestamp=t)
            assert not triggered
            if triggered:
                driver.left_click()

        assert driver.left_clicks == 1  # Continua 1

        # 3. Afastamento para rearmar (> 45 px)
        t = clock.advance(0.0333)
        dwell.update(560.0, 500.0, timestamp=t)  # 60 px de distância

        # 4. Nova fixação de 950 ms na nova posição
        click_fired_2 = False
        for _ in range(30):
            t = clock.advance(0.0333)
            triggered, pct, _ = dwell.update(560.0, 500.0, timestamp=t)
            if triggered:
                click_fired_2 = True
                driver.left_click()

        assert click_fired_2 is True
        assert driver.left_clicks == 2

    def test_drag_and_drop_state_lifecycle_integration(self):
        """
        Valida que o estado de arraste (drag) iniciado por gesto é desarmado
        imediatamente se o rastreamento for perdido ou se a aplicação for pausada.
        """
        driver = MockMouseDriver()
        mouse = MouseController(backend=driver)
        state_machine = StateMachine(
            initial=AppState.ACTIVE,
            on_release_all=mouse.release_all,
        )

        # Inicia arraste
        mouse.start_drag()
        assert mouse.is_dragging is True
        assert driver._left_down is True
        assert "left" in driver.button_downs

        # Simula movimentação durante arraste
        mouse.move(300, 400, timestamp=1.0)
        mouse.move(350, 450, timestamp=1.033)
        assert mouse.is_dragging is True

        # Perda de rastreamento (ACTIVE -> TRACKING_LOST)
        state_machine.try_transition(AppState.TRACKING_LOST)

        # O callback on_release_all DEVE ter desarmado o mouse no driver
        assert driver._left_down is False
        assert "left" in driver.button_ups
        assert mouse.is_dragging is False

    def test_bilateral_blink_arbitration_integration(self):
        """
        Verifica que piscadas bilaterais involuntárias (ambos os olhos fechando simultaneamente
        ou em janela <= 80ms) são descartadas pelo GestureEngine.
        """
        engine = GestureEngine(click_freeze_sec=0.150, bilateral_window_sec=0.080, enable_double_blink=False)
        clock = SimulatedClock(10.0)

        # 1. Ambos os olhos fecham juntos no mesmo frame -> descartado
        t1 = clock.now()
        action1 = engine.process_blink_events(
            left_blink=True, right_blink=True, double_blink=False,
            hold_start=False, hold_end=False, timestamp=t1
        )
        assert action1 is None

        # 2. Olho esquerdo pisca primeiro em t=11.0, e olho direito pisca 40ms depois (dentro da janela de 80ms)
        t_left = clock.advance(1.0)
        action_left = engine.process_blink_events(
            left_blink=True, right_blink=False, double_blink=False,
            hold_start=False, hold_end=False, timestamp=t_left
        )
        # Olho direito pisca logo a seguir
        t_right = clock.advance(0.040)
        action_right = engine.process_blink_events(
            left_blink=False, right_blink=True, double_blink=False,
            hold_start=False, hold_end=False, timestamp=t_right
        )
        assert action_right is None

    def test_unilateral_blink_with_click_freeze(self):
        """
        Verifica que uma piscada unilateral isolada dispara o clique esquerdo
        e restaura a posição antes da oclusão palpebral (Click Freeze).
        """
        engine = GestureEngine(click_freeze_sec=0.150, bilateral_window_sec=0.080)
        clock = SimulatedClock(20.0)

        # Alimenta buffer de histórico de posições com coordenadas pré-oclusão
        for i in range(15):
            t = clock.advance(0.020)
            engine.record_stable_position(100.0 + i, 200.0 + i, timestamp=t)

        # Em t=20.30, ocorre piscada unilateral do olho esquerdo
        t_blink = clock.now()
        action = engine.process_blink_events(
            left_blink=True, right_blink=False, double_blink=False,
            hold_start=False, hold_end=False, timestamp=t_blink
        )
        assert action == MouseAction.LEFT_CLICK

        # As coordenadas estabilizadas pelo Click Freeze devem refletir o buffer pré-oclusão (~120ms atrás)
        # O cursor atual no instante do clique seria (114, 214), mas a posição congelada é anterior
        frozen_pos = engine.get_stabilized_position((114.0, 214.0), timestamp=t_blink)
        assert frozen_pos[0] < 114.0
        assert frozen_pos[1] < 214.0

    def test_resolution_adaptation_integration(self):
        """
        Verifica que a adaptação dinâmica de resolução ajusta as predições
        proporcionalmente sem necessidade de recalibração completa.
        """
        calib = CalibrationManager(profile_name="res_test")
        calib.model = PolynomialCalibrationModel()
        calib.model.coeffs_x = np.array([0.0, 1920.0, 0.0, 0.0, 0.0, 0.0])
        calib.model.coeffs_y = np.array([0.0, 0.0, 1080.0, 0.0, 0.0, 0.0])
        calib.model.is_fitted = True
        calib.is_calibrated = True
        calib.calibrated_screen_w = 1920
        calib.calibrated_screen_h = 1080

        # Resolução original (1920x1080) -> íris (0.5, 0.5) mapeia para centro (960, 540)
        pos_orig = calib.map_to_screen((0.5, 0.5))
        assert pos_orig == (960, 540)

        # Altera resolução do Windows para 2560x1440
        adapted = calib.adapt_screen_resolution(2560, 1440)
        assert adapted is True
        assert calib.is_resolution_compatible(2560, 1440) is False

        # Nova predição no centro deve ser exatamente (1280, 720)
        pos_adapted = calib.map_to_screen((0.5, 0.5))
        assert pos_adapted == (1280, 720)

    def test_precision_mode_attenuation(self):
        """
        Verifica que no modo de precisão, pequenos movimentos são atenuados (ganho 35%).
        """
        driver = MockMouseDriver()
        mouse = MouseController(backend=driver, filter_type="NONE")

        # Posição neutra
        mouse.move(500, 500, timestamp=1.0)
        assert driver.moves[-1] == (500, 500)

        # Ativa modo de precisão (fator 0.35)
        mouse.set_precision_mode(True, factor=0.35)

        # Deslocamento bruto de 100 pixels em X (de 500 para 600)
        mouse.move(600, 500, timestamp=1.033)
        # O movimento efetivo deve ser 500 + 100 * 0.35 = 535
        assert driver.moves[-1] == (535, 500)

        # Desativa modo de precisão
        mouse.set_precision_mode(False)
        mouse.move(700, 500, timestamp=1.066)
        # Agora o movimento é 100% normal
        assert driver.moves[-1] == (700, 500)
