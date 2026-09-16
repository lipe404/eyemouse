"""
test_settings_manager.py — Testes unitários para SettingsManager (Milestone 6).

Testa:
  - Criação de diretórios e perfis padrão.
  - Validação defensiva de esquema (tipagem, limites, chaves ausentes).
  - Escrita atômica e prevenção de corrupção.
  - Sanitização de nomes de perfis (prevenção de path traversal).
  - Limpeza completa de dados do usuário (privacidade garantida).
"""
import os
import json
import pytest
from eye_mouse.settings_manager import SettingsManager


@pytest.fixture
def temp_settings_manager(tmp_path):
    """Cria uma instância isolada de SettingsManager apontando para tmp_path."""
    manager = SettingsManager(base_dir=str(tmp_path))
    return manager


class TestSettingsManagerDefaultsAndValidation:
    def test_default_settings_schema(self, temp_settings_manager):
        defaults = temp_settings_manager.get_default_settings()
        assert "camera" in defaults
        assert "interaction" in defaults
        assert "smoothing" in defaults
        assert "gestures" in defaults
        assert "accessibility" in defaults
        assert "calibration" in defaults
        assert defaults["interaction"]["profile"].upper() == "HYBRID"
        assert defaults["smoothing"]["filter_type"].upper() == "ONE_EURO"

    def test_validate_settings_fills_missing_keys(self, temp_settings_manager):
        partial = {
            "camera": {"camera_index": 1},
            "interaction": {"profile": "DWELL"}
        }
        validated = temp_settings_manager.validate_settings(partial)
        assert validated["camera"]["index"] == 1
        assert validated["camera"]["width"] == 640
        assert validated["interaction"]["profile"] == "DWELL"
        assert validated["smoothing"]["filter_type"] == "ONE_EURO"
        assert validated["accessibility"]["show_action_bar"] is True

    def test_validate_settings_clamps_bounds(self, temp_settings_manager):
        out_of_bounds = {
            "camera": {"camera_index": -5, "fps": 240},
            "interaction": {"profile": "invalid_profile", "dwell_time_sec": 0.01, "dwell_radius_pixels": 500},
            "smoothing": {"one_euro_min_cutoff": -1.0},
            "gestures": {"ear_blink_threshold": 0.99, "click_freeze_duration_sec": -0.5},
        }
        validated = temp_settings_manager.validate_settings(out_of_bounds)
        assert validated["camera"]["index"] == 0
        assert validated["camera"]["fps"] == 120
        assert validated["interaction"]["profile"] == "HYBRID"  # fallback
        assert validated["interaction"]["dwell_time_sec"] >= 0.2
        assert validated["interaction"]["dwell_radius_px"] <= 150.0
        assert validated["gestures"]["blink_ear_threshold"] <= 0.50
        assert validated["gestures"]["hysteresis_margin"] >= 0.005


class TestSettingsManagerPersistence:
    def test_load_non_existent_profile_returns_default(self, temp_settings_manager):
        profile = temp_settings_manager.load_profile("non_existent_user")
        assert profile["interaction"]["profile"].upper() == "HYBRID"

    def test_save_and_load_profile_roundtrip(self, temp_settings_manager):
        user = "alice"
        data = temp_settings_manager.get_default_settings()
        data["camera"]["index"] = 2
        data["interaction"]["profile"] = "DWELL"

        saved = temp_settings_manager.save_profile(user, data)
        assert saved is True

        loaded = temp_settings_manager.load_profile(user)
        assert loaded["camera"]["index"] == 2
        assert loaded["interaction"]["profile"] == "DWELL"

    def test_profile_path_traversal_sanitized(self, temp_settings_manager):
        profile = temp_settings_manager.load_profile("../../malicious")
        assert profile is not None
        # Sem caracteres alfanuméricos deve cair em 'default'
        saved = temp_settings_manager.save_profile("../../..", {"camera": {"index": 1}})
        assert saved is True
        assert os.path.exists(temp_settings_manager.get_profile_path("default"))

    def test_corrupted_json_handling(self, temp_settings_manager):
        path = temp_settings_manager.get_profile_path("corrupted_user")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ invalid json : [")

        loaded = temp_settings_manager.load_profile("corrupted_user")
        assert loaded["interaction"]["profile"].upper() == "HYBRID"

    def test_list_and_delete_profiles(self, temp_settings_manager):
        temp_settings_manager.save_profile("user1", temp_settings_manager.get_default_settings())
        temp_settings_manager.save_profile("user2", temp_settings_manager.get_default_settings())

        profiles = temp_settings_manager.list_profiles()
        assert "user1" in profiles
        assert "user2" in profiles

        deleted = temp_settings_manager.delete_profile("user1")
        assert deleted is True
        profiles_after = temp_settings_manager.list_profiles()
        assert "user1" not in profiles_after
        assert "user2" in profiles_after


class TestSettingsManagerPrivacy:
    def test_clear_all_user_data(self, temp_settings_manager, tmp_path):
        temp_settings_manager.save_profile("bob", temp_settings_manager.get_default_settings())
        calib_dir = os.path.join(str(tmp_path), "calibrations")
        os.makedirs(calib_dir, exist_ok=True)
        with open(os.path.join(calib_dir, "bob_calib.json"), "w") as f:
            f.write("{}")

        temp_settings_manager.clear_all_user_data()

        profiles = temp_settings_manager.list_profiles()
        assert len(profiles) == 0
        assert not os.path.exists(os.path.join(calib_dir, "bob_calib.json"))
