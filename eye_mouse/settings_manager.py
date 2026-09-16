"""
settings_manager.py — Gerenciador de configurações persistentes e privacidade do usuário (Milestone 6).

Responsabilidades:
  - Persistência estruturada e validada de configurações em formato JSON.
  - Armazenamento em diretório do usuário: ~/Documents/EyeMouse/profiles/<nome>.json.
  - Validação estrita de schema, limites e tipos com fallbacks defensivos.
  - Privacidade por design:
      * Armazena exclusivamente parâmetros de configuração.
      * NUNCA armazena frames de vídeo, imagens da webcam ou biometria facial.
      * Fornece métodos explícitos para exclusão de dados e restauração de fábrica.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from config import (
    BLINK_EAR_THRESHOLD,
    CAMERA_BACKEND,
    CAMERA_FOURCC,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CLICK_FREEZE_DURATION_SEC,
    DWELL_RADIUS_PIXELS,
    DWELL_TIME_SEC,
    INTERACTION_PROFILE,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_MIN_CUTOFF,
    PRECISION_MODE_FACTOR,
    SMOOTHING_FILTER_TYPE,
    TARGET_FPS,
    USER_DATA_DIR,
)

logger = logging.getLogger(__name__)

PROFILES_DIR = os.path.join(USER_DATA_DIR, "profiles")


def get_default_settings() -> Dict[str, Any]:
    """Retorna as configurações padrão do sistema com validação de tipos."""
    return {
        "version": 1,
        "camera": {
            "index": CAMERA_INDEX,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fps": TARGET_FPS,
            "backend": CAMERA_BACKEND,
            "fourcc": CAMERA_FOURCC,
        },
        "interaction": {
            "profile": INTERACTION_PROFILE,  # "HYBRID", "DWELL", "CONTINUOUS"
            "dwell_time_sec": DWELL_TIME_SEC,
            "dwell_radius_px": DWELL_RADIUS_PIXELS,
            "scroll_enabled": True,
            "precision_factor": PRECISION_MODE_FACTOR,
            "click_freeze_sec": CLICK_FREEZE_DURATION_SEC,
        },
        "smoothing": {
            "filter_type": SMOOTHING_FILTER_TYPE,  # "ONE_EURO", "KALMAN", "NONE"
            "one_euro_min_cutoff": ONE_EURO_MIN_CUTOFF,
            "one_euro_beta": ONE_EURO_BETA,
            "one_euro_d_cutoff": ONE_EURO_D_CUTOFF,
        },
        "gestures": {
            "enable_left_blink": True,
            "enable_right_blink": True,
            "enable_double_blink": False,
            "enable_hold_drag": True,
            "blink_ear_threshold": BLINK_EAR_THRESHOLD,
            "hysteresis_margin": 0.020,
        },
        "accessibility": {
            "audio_feedback": True,
            "visual_dwell_indicator": True,
            "animations_enabled": True,
            "high_contrast": False,
            "show_action_bar": True,
        },
        "calibration": {
            "model_type": "polynomial_2nd",
            "holdout_error": 0.0,
            "is_calibrated": False,
        },
    }


class SettingsManager:
    """
    Gerencia o ciclo de vida e a persistência de configurações do usuário.
    """

    def __init__(self, profiles_dir: Optional[str] = None, base_dir: Optional[str] = None):
        if base_dir:
            self.user_data_dir = base_dir
            self.profiles_dir = profiles_dir or os.path.join(base_dir, "profiles")
        else:
            self.user_data_dir = USER_DATA_DIR
            self.profiles_dir = profiles_dir or PROFILES_DIR
        os.makedirs(self.profiles_dir, exist_ok=True)

    def get_profile_path(self, profile_name: str) -> str:
        return self._get_profile_path(profile_name)

    def get_default_settings(self) -> Dict[str, Any]:
        return get_default_settings()

    def list_profiles(self) -> List[str]:
        """Retorna os nomes de todos os perfis existentes."""
        if not os.path.exists(self.profiles_dir):
            return []
        profiles = []
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith(".json") and not fname.endswith(".tmp"):
                profiles.append(fname[:-5])
        return sorted(profiles)

    def _get_profile_path(self, profile_name: str) -> str:
        safe_name = "".join(c for c in profile_name if c.isalnum() or c in ("-", "_")).strip()
        if not safe_name:
            safe_name = "default"
        return os.path.join(self.profiles_dir, f"{safe_name}.json")

    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida a estrutura e os limites numéricos das configurações, aplicando defaults em caso de inconsistência.
        """
        defaults = get_default_settings()
        validated: Dict[str, Any] = {}

        validated["version"] = int(settings.get("version", defaults["version"]))

        # 1. Câmera
        cam_in = settings.get("camera", {})
        cam_def = defaults["camera"]
        cam_idx = cam_in.get("index", cam_in.get("camera_index", cam_def["index"]))
        validated["camera"] = {
            "index": max(0, int(cam_idx)),
            "width": max(320, min(3840, int(cam_in.get("width", cam_def["width"])))),
            "height": max(240, min(2160, int(cam_in.get("height", cam_def["height"])))),
            "fps": max(15, min(120, int(cam_in.get("fps", cam_def["fps"])))),
            "backend": str(cam_in.get("backend", cam_def["backend"])),
            "fourcc": str(cam_in.get("fourcc", cam_def["fourcc"])),
        }

        # 2. Interação
        inter_in = settings.get("interaction", {})
        inter_def = defaults["interaction"]
        p_raw = str(inter_in.get("profile", inter_def["profile"])).upper()
        p_val = p_raw if p_raw in ("CONTINUOUS", "DWELL", "HYBRID") else inter_def["profile"]
        dw_rad = inter_in.get("dwell_radius_px", inter_in.get("dwell_radius_pixels", inter_def["dwell_radius_px"]))
        clk_frz = inter_in.get("click_freeze_sec", inter_in.get("click_freeze_duration_sec", inter_def["click_freeze_sec"]))

        validated["interaction"] = {
            "profile": p_val,
            "dwell_time_sec": max(0.2, min(3.0, float(inter_in.get("dwell_time_sec", inter_def["dwell_time_sec"])))),
            "dwell_radius_px": max(10.0, min(150.0, float(dw_rad))),
            "scroll_enabled": bool(inter_in.get("scroll_enabled", inter_def["scroll_enabled"])),
            "precision_factor": max(0.05, min(1.0, float(inter_in.get("precision_factor", inter_def["precision_factor"])))),
            "click_freeze_sec": max(0.05, min(0.5, float(clk_frz))),
        }

        # 3. Suavização
        smooth_in = settings.get("smoothing", {})
        smooth_def = defaults["smoothing"]
        f_raw = str(smooth_in.get("filter_type", smooth_def["filter_type"])).upper()
        f_val = f_raw if f_raw in ("ONE_EURO", "KALMAN", "NONE") else smooth_def["filter_type"]

        validated["smoothing"] = {
            "filter_type": f_val,
            "one_euro_min_cutoff": max(0.01, min(10.0, float(smooth_in.get("one_euro_min_cutoff", smooth_def["one_euro_min_cutoff"])))),
            "one_euro_beta": max(0.0, min(1.0, float(smooth_in.get("one_euro_beta", smooth_def["one_euro_beta"])))),
            "one_euro_d_cutoff": max(0.01, min(10.0, float(smooth_in.get("one_euro_d_cutoff", smooth_def["one_euro_d_cutoff"])))),
        }

        # 4. Gestos
        gest_in = settings.get("gestures", {})
        gest_def = defaults["gestures"]
        blk_thresh = gest_in.get("blink_ear_threshold", gest_in.get("ear_blink_threshold", gest_def["blink_ear_threshold"]))
        validated["gestures"] = {
            "enable_left_blink": bool(gest_in.get("enable_left_blink", gest_def["enable_left_blink"])),
            "enable_right_blink": bool(gest_in.get("enable_right_blink", gest_def["enable_right_blink"])),
            "enable_double_blink": bool(gest_in.get("enable_double_blink", gest_def["enable_double_blink"])),
            "enable_hold_drag": bool(gest_in.get("enable_hold_drag", gest_def["enable_hold_drag"])),
            "blink_ear_threshold": max(0.05, min(0.50, float(blk_thresh))),
            "hysteresis_margin": max(0.005, min(0.08, float(gest_in.get("hysteresis_margin", gest_def["hysteresis_margin"])))),
        }

        # 5. Acessibilidade
        acc_in = settings.get("accessibility", {})
        acc_def = defaults["accessibility"]
        validated["accessibility"] = {
            "audio_feedback": bool(acc_in.get("audio_feedback", acc_def["audio_feedback"])),
            "visual_dwell_indicator": bool(acc_in.get("visual_dwell_indicator", acc_def["visual_dwell_indicator"])),
            "animations_enabled": bool(acc_in.get("animations_enabled", acc_def["animations_enabled"])),
            "high_contrast": bool(acc_in.get("high_contrast", acc_def["high_contrast"])),
            "show_action_bar": bool(acc_in.get("show_action_bar", acc_def["show_action_bar"])),
        }

        # 6. Calibração
        cal_in = settings.get("calibration", {})
        cal_def = defaults["calibration"]
        validated["calibration"] = {
            "model_type": str(cal_in.get("model_type", cal_def["model_type"])),
            "holdout_error": max(0.0, float(cal_in.get("holdout_error", cal_def["holdout_error"]))),
            "is_calibrated": bool(cal_in.get("is_calibrated", cal_def["is_calibrated"])),
        }

        return validated

    def load_profile(self, profile_name: str = "default") -> Dict[str, Any]:
        """
        Carrega as configurações salvas do usuário. Se o arquivo não existir ou for inválido,
        retorna as configurações padrão seguras.
        """
        path = self._get_profile_path(profile_name)
        if not os.path.exists(path):
            logger.info("Perfil '%s' não encontrado. Usando padrões.", profile_name)
            return get_default_settings()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.validate_settings(data)
        except Exception as exc:
            logger.warning("Erro ao carregar perfil '%s' (%s). Usando padrões.", profile_name, exc)
            return get_default_settings()

    def save_profile(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        """
        Salva as configurações do usuário de forma atômica e validada.
        """
        validated = self.validate_settings(settings)
        path = self._get_profile_path(profile_name)
        tmp_path = path + ".tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
            logger.info("Configurações do perfil '%s' salvas com sucesso em %s", profile_name, path)
            return True
        except Exception as exc:
            logger.error("Erro ao salvar perfil '%s': %s", profile_name, exc)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False

    def delete_profile(self, profile_name: str) -> bool:
        """
        Remove todas as configurações e calibrações associadas ao perfil indicado.
        """
        path = self._get_profile_path(profile_name)
        deleted = False

        if os.path.exists(path):
            try:
                os.remove(path)
                deleted = True
                logger.info("Perfil '%s' removido com sucesso.", profile_name)
            except Exception as exc:
                logger.error("Erro ao excluir arquivo de perfil '%s': %s", path, exc)

        # Exclui também arquivos de calibração associados
        cal_path = os.path.join(self.user_data_dir, f"calibration_{profile_name}.json")
        if os.path.exists(cal_path):
            try:
                os.remove(cal_path)
                deleted = True
                logger.info("Arquivo de calibração '%s' removido.", cal_path)
            except Exception as exc:
                logger.error("Erro ao remover calibração associada: %s", exc)

        return deleted

    def clear_all_user_data(self) -> bool:
        """
        Exclui todos os perfis, calibrações e dados locais do usuário para garantia de privacidade.
        """
        try:
            if os.path.exists(self.profiles_dir):
                for fname in os.listdir(self.profiles_dir):
                    fpath = os.path.join(self.profiles_dir, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)

            if os.path.exists(self.user_data_dir):
                for fname in os.listdir(self.user_data_dir):
                    if (fname.startswith("calibration_") or fname.endswith("_calib.json")) and fname.endswith(".json"):
                        fpath = os.path.join(self.user_data_dir, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)

                calib_sub = os.path.join(self.user_data_dir, "calibrations")
                if os.path.exists(calib_sub):
                    for fname in os.listdir(calib_sub):
                        fpath = os.path.join(calib_sub, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)

            logger.info("Todos os dados do usuário e calibrações foram apagados com sucesso.")
            return True
        except Exception as exc:
            logger.error("Falha ao apagar dados do usuário: %s", exc)
            return False
