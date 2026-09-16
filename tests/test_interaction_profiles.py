"""
test_interaction_profiles.py — Testes dos Perfis de Interação e Acessibilidade (Milestone 5).

Verifica:
  - Perfis de fábrica: CONTINUOUS, DWELL, HYBRID
  - Princípio de Acessibilidade: requires_asymmetric_blink é False em todos os perfis
  - DWELL desabilita gestos oculares voluntários
  - Customização de perfis e sincronização
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from interaction_profiles import (
    InteractionProfileType,
    ProfileManager,
    DEFAULT_PROFILES,
)


class TestInteractionProfiles:
    """Testes do gerenciador de perfis de interação."""

    def test_default_profiles_exist(self):
        assert InteractionProfileType.CONTINUOUS in DEFAULT_PROFILES
        assert InteractionProfileType.DWELL in DEFAULT_PROFILES
        assert InteractionProfileType.HYBRID in DEFAULT_PROFILES

    def test_accessibility_principle_no_mandatory_asymmetric_blink(self):
        """Nenhum perfil de fábrica pode exigir piscada assimétrica para funcionalidades vitais."""
        for ptype, cfg in DEFAULT_PROFILES.items():
            assert cfg.requires_asymmetric_blink is False, f"Perfil {ptype.name} violou acessibilidade!"

    def test_dwell_profile_disables_voluntary_blinks(self):
        """O perfil DWELL desabilita piscadas para evitar acionamentos acidentais."""
        cfg = DEFAULT_PROFILES[InteractionProfileType.DWELL]
        assert cfg.enable_left_blink is False
        assert cfg.enable_right_blink is False
        assert cfg.enable_dwell_click is True
        assert cfg.enable_action_bar is True

    def test_continuous_profile_enables_blinks_disables_dwell(self):
        cfg = DEFAULT_PROFILES[InteractionProfileType.CONTINUOUS]
        assert cfg.enable_left_blink is True
        assert cfg.enable_right_blink is True
        assert cfg.enable_dwell_click is False

    def test_profile_manager_switching(self):
        mgr = ProfileManager(initial_profile="HYBRID")
        assert mgr.current_profile_type == InteractionProfileType.HYBRID

        # Muda para Dwell
        cfg = mgr.set_profile("DWELL")
        assert mgr.current_profile_type == InteractionProfileType.DWELL
        assert cfg.enable_dwell_click is True

        # Aceita nomes normalizados em português
        mgr.set_profile("contínuo")
        assert mgr.current_profile_type == InteractionProfileType.CONTINUOUS

    def test_customization_preserves_independence(self):
        mgr = ProfileManager(initial_profile="HYBRID")
        # Desabilita piscada direita no híbrido
        mgr.customize_gesture(enable_right_blink=False)
        assert mgr.active_config.enable_right_blink is False

        # O perfil padrão de fábrica original não deve ter sido corrompido
        assert DEFAULT_PROFILES[InteractionProfileType.HYBRID].enable_right_blink is True
