"""
test_blink_detector_m5.py — Testes da máquina de estados de piscada e histerese (Milestone 5).
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from blink_detector import BlinkDetector, EyeBlinkState


class TestBlinkDetectorM5StateMachine:
    """Valida a máquina de estados explícita do BlinkDetector."""

    def test_initial_state_is_eyes_open(self):
        detector = BlinkDetector()
        assert detector.left_state == EyeBlinkState.EYES_OPEN
        assert detector.right_state == EyeBlinkState.EYES_OPEN

    def test_hysteresis_thresholds(self):
        detector = BlinkDetector(ear_threshold=0.20, hysteresis_margin=0.03)
        assert detector.ear_close_threshold == pytest.approx(0.17)
        assert detector.ear_open_threshold == pytest.approx(0.23)

    def test_transition_open_to_possible_closing(self):
        detector = BlinkDetector(ear_threshold=0.20, hysteresis_margin=0.03)
        now = time.perf_counter()

        # EAR abaixo do limiar de fechamento (0.17)
        detector.update_eye_state_for_test("left", ear=0.15, now=now)
        assert detector.left_state == EyeBlinkState.POSSIBLE_CLOSING

    def test_transition_to_closure_confirmed(self):
        detector = BlinkDetector(ear_threshold=0.20, hysteresis_margin=0.03)
        t0 = 100.0

        # Inicia fechamento
        detector.update_eye_state_for_test("left", ear=0.15, now=t0)
        assert detector.left_state == EyeBlinkState.POSSIBLE_CLOSING

        # Permanece fechado no próximo frame
        detector.update_eye_state_for_test("left", ear=0.14, now=t0 + 0.080)
        assert detector.left_state == EyeBlinkState.CLOSURE_CONFIRMED

    def test_hysteresis_prevents_chatter(self):
        """Entre 0.17 e 0.23, o estado deve permanecer inalterado (sem flutter)."""
        detector = BlinkDetector(ear_threshold=0.20, hysteresis_margin=0.03)
        t0 = 100.0

        # Olho aberto com EAR 0.25
        detector.update_eye_state_for_test("left", ear=0.25, now=t0)
        assert detector.left_state == EyeBlinkState.EYES_OPEN

        # EAR cai para 0.19 (zona neutra de histerese, ainda acima de 0.17)
        detector.update_eye_state_for_test("left", ear=0.19, now=t0 + 0.033)
        assert detector.left_state == EyeBlinkState.EYES_OPEN

    def test_complete_voluntary_blink_triggers_gesture(self):
        """Piscada voluntária típica conclui o gesto e retorna action_triggered=True."""
        detector = BlinkDetector(ear_threshold=0.20, hysteresis_margin=0.03)
        t0 = 100.0

        # 1. Olhos abertos
        g, hs, he = detector.update_eye_state_for_test("left", ear=0.25, now=t0)
        assert not g

        # 2. Início do fechamento
        g, hs, he = detector.update_eye_state_for_test("left", ear=0.15, now=t0 + 0.033)
        assert not g

        # 3. Confirmação do fechamento
        g, hs, he = detector.update_eye_state_for_test("left", ear=0.14, now=t0 + 0.100)
        assert not g

        # 4. Olho reabre com EAR 0.25 com derivada suave (voluntária) após 180ms total
        # deriv=0.04 (abaixo de BLINK_REFLEX_OPEN_SPEED=0.08)
        g, hs, he = detector.update_eye_state_for_test("left", ear=0.25, now=t0 + 0.180, deriv=0.04)

        assert g is True
        assert detector.left_state == EyeBlinkState.GESTURE_COMPLETED

    def test_hold_drag_detection(self):
        """Manter o olho fechado por > 1.5s ativa hold_start e reabertura ativa hold_end."""
        detector = BlinkDetector(ear_threshold=0.20, hysteresis_margin=0.03)
        t0 = 100.0

        # Inicia fechamento
        detector.update_eye_state_for_test("left", ear=0.12, now=t0)
        # Confirma fechamento
        detector.update_eye_state_for_test("left", ear=0.12, now=t0 + 0.100)

        # Permanece fechado por 1.6 segundos -> hold_start
        g, hs, he = detector.update_eye_state_for_test("left", ear=0.12, now=t0 + 1.600)
        assert hs is True
        assert g is False

        # Reabre o olho -> hold_end
        g, hs, he = detector.update_eye_state_for_test("left", ear=0.26, now=t0 + 1.800)
        assert he is True

    def test_independent_eye_references(self):
        """Valida referências separadas de olho esquerdo e direito."""
        detector = BlinkDetector()
        detector.set_eye_references(
            left_open=0.28, left_closed=0.12,
            right_open=0.32, right_closed=0.14
        )
        assert detector.left_open_ear == 0.28
        assert detector.left_closed_ear == 0.12
        assert detector.right_open_ear == 0.32
        assert detector.right_closed_ear == 0.14
