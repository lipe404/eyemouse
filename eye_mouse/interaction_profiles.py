"""
interaction_profiles.py — Perfis de interação e acessibilidade universal (Milestone 5).

Define e gerencia os perfis de uso do EyeMouse:
  - CONTINUOUS (Contínuo):
      O cursor acompanha o olhar e os comandos são disparados por gestos voluntários (piscadas).
  - DWELL (Permanência):
      O cursor acompanha o olhar e o clique ocorre por fixação estável (Dwell Click).
      Ideal para usuários que não conseguem ou não desejam piscar voluntariamente.
  - HYBRID (Híbrido):
      Combina rastreamento contínuo, dwell na barra de ações e suporte a gestos e atalhos.

Princípio Fundamental de Acessibilidade:
  Nenhuma função essencial (clique direito, duplo clique, arraste, rolagem) exige que o usuário
  consiga piscar apenas um dos olhos isoladamente.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class InteractionProfileType(Enum):
    """Tipos de perfil de interação disponíveis."""
    CONTINUOUS = "CONTINUOUS"
    DWELL = "DWELL"
    HYBRID = "HYBRID"


@dataclass
class ProfileConfig:
    """Configuração de recursos e gestos por perfil."""
    profile_type: InteractionProfileType
    description: str
    enable_left_blink: bool
    enable_right_blink: bool
    enable_double_blink: bool
    enable_hold_drag: bool
    enable_dwell_click: bool
    enable_action_bar: bool
    enable_scroll_mode: bool
    requires_asymmetric_blink: bool = False


# Perfis pré-definidos de fábrica
DEFAULT_PROFILES: Dict[InteractionProfileType, ProfileConfig] = {
    InteractionProfileType.CONTINUOUS: ProfileConfig(
        profile_type=InteractionProfileType.CONTINUOUS,
        description="Cursor contínuo com comandos por gestos oculares voluntários e atalhos.",
        enable_left_blink=True,
        enable_right_blink=True,
        enable_double_blink=False,
        enable_hold_drag=True,
        enable_dwell_click=False,
        enable_action_bar=True,
        enable_scroll_mode=True,
        requires_asymmetric_blink=False,
    ),
    InteractionProfileType.DWELL: ProfileConfig(
        profile_type=InteractionProfileType.DWELL,
        description="Controle por permanência do olhar (Dwell). Dispensa qualquer piscada voluntária.",
        enable_left_blink=False,
        enable_right_blink=False,
        enable_double_blink=False,
        enable_hold_drag=False,
        enable_dwell_click=True,
        enable_action_bar=True,
        enable_scroll_mode=True,
        requires_asymmetric_blink=False,
    ),
    InteractionProfileType.HYBRID: ProfileConfig(
        profile_type=InteractionProfileType.HYBRID,
        description="Perfil flexível combinando fixação (dwell), barra flutuante e gestos opcionais.",
        enable_left_blink=True,
        enable_right_blink=True,
        enable_double_blink=False,
        enable_hold_drag=True,
        enable_dwell_click=True,
        enable_action_bar=True,
        enable_scroll_mode=True,
        requires_asymmetric_blink=False,
    ),
}


class ProfileManager:
    """
    Gerenciador de perfis de interação ativos.
    """

    def __init__(self, initial_profile: str = "HYBRID"):
        self._current_profile_type: InteractionProfileType = self._parse_profile_type(initial_profile)
        self._custom_configs: Dict[InteractionProfileType, ProfileConfig] = {
            k: ProfileConfig(**v.__dict__) for k, v in DEFAULT_PROFILES.items()
        }

    def _parse_profile_type(self, name: str) -> InteractionProfileType:
        norm = name.upper().strip()
        if norm in ("CONTINUOUS", "CONTÍNUO", "CONTINUO"):
            return InteractionProfileType.CONTINUOUS
        elif norm in ("DWELL", "PERMANÊNCIA", "PERMANENCIA"):
            return InteractionProfileType.DWELL
        elif norm in ("HYBRID", "HÍBRIDO", "HIBRIDO"):
            return InteractionProfileType.HYBRID
        else:
            logger.warning("Perfil desconhecido '%s'. Usando HYBRID como padrão.", name)
            return InteractionProfileType.HYBRID

    @property
    def current_profile_type(self) -> InteractionProfileType:
        return self._current_profile_type

    @property
    def active_config(self) -> ProfileConfig:
        return self._custom_configs[self._current_profile_type]

    def set_profile(self, profile_name: str) -> ProfileConfig:
        """Altera o perfil ativo do sistema."""
        self._current_profile_type = self._parse_profile_type(profile_name)
        cfg = self.active_config
        logger.info("ProfileManager: perfil ativo alterado para %s (%s)", cfg.profile_type.name, cfg.description)
        return cfg

    def customize_gesture(
        self,
        profile_type: Optional[InteractionProfileType] = None,
        enable_left_blink: Optional[bool] = None,
        enable_right_blink: Optional[bool] = None,
        enable_double_blink: Optional[bool] = None,
        enable_hold_drag: Optional[bool] = None,
        enable_dwell_click: Optional[bool] = None,
        enable_action_bar: Optional[bool] = None,
        enable_scroll_mode: Optional[bool] = None,
    ) -> ProfileConfig:
        """Personaliza individualmente recursos do perfil."""
        pt = profile_type or self._current_profile_type
        cfg = self._custom_configs[pt]

        if enable_left_blink is not None: cfg.enable_left_blink = bool(enable_left_blink)
        if enable_right_blink is not None: cfg.enable_right_blink = bool(enable_right_blink)
        if enable_double_blink is not None: cfg.enable_double_blink = bool(enable_double_blink)
        if enable_hold_drag is not None: cfg.enable_hold_drag = bool(enable_hold_drag)
        if enable_dwell_click is not None: cfg.enable_dwell_click = bool(enable_dwell_click)
        if enable_action_bar is not None: cfg.enable_action_bar = bool(enable_action_bar)
        if enable_scroll_mode is not None: cfg.enable_scroll_mode = bool(enable_scroll_mode)

        return cfg
