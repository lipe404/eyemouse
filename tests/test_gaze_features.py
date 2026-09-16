"""
test_gaze_features.py — Testes das features oculares relativas, pose da cabeça e fusão.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from gaze_features import (
    GazeFeatureExtractor,
    EyeMetrics,
    HeadPose,
    CombinedGazeFeatures,
    build_feature_vector,
    LEFT_INNER_CANTHUS,
    LEFT_OUTER_CANTHUS,
    LEFT_UPPER_EYELID,
    LEFT_LOWER_EYELID,
    LEFT_IRIS_ALL,
    RIGHT_INNER_CANTHUS,
    RIGHT_OUTER_CANTHUS,
    RIGHT_UPPER_EYELID,
    RIGHT_LOWER_EYELID,
    RIGHT_IRIS_ALL,
    NOSE_TIP,
    CHIN,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)


def create_synthetic_landmarks(
    face_offset_x: float = 0.0,
    face_offset_y: float = 0.0,
    left_iris_offset_x: float = 0.0,
    left_closed: bool = False,
    right_closed: bool = False,
) -> list:
    """Cria lista de 500 landmarks sintéticos com posições conhecidas."""
    lms = [MagicMock(x=0.5 + face_offset_x, y=0.5 + face_offset_y, z=0.0) for _ in range(500)]

    # Olho esquerdo (canthus externo = 33, interno = 133)
    lms[LEFT_OUTER_CANTHUS].x = 0.30 + face_offset_x
    lms[LEFT_OUTER_CANTHUS].y = 0.35 + face_offset_y
    lms[LEFT_INNER_CANTHUS].x = 0.40 + face_offset_x
    lms[LEFT_INNER_CANTHUS].y = 0.35 + face_offset_y

    # Pálpebras esquerda
    eyelid_gap_l = 0.005 if left_closed else 0.035
    lms[LEFT_UPPER_EYELID].x = 0.35 + face_offset_x
    lms[LEFT_UPPER_EYELID].y = 0.35 - (eyelid_gap_l / 2.0) + face_offset_y
    lms[LEFT_LOWER_EYELID].x = 0.35 + face_offset_x
    lms[LEFT_LOWER_EYELID].y = 0.35 + (eyelid_gap_l / 2.0) + face_offset_y

    # Íris esquerda
    for idx in LEFT_IRIS_ALL:
        lms[idx].x = 0.35 + left_iris_offset_x + face_offset_x
        lms[idx].y = 0.35 + face_offset_y

    # Olho direito (interno = 362, externo = 263)
    lms[RIGHT_INNER_CANTHUS].x = 0.60 + face_offset_x
    lms[RIGHT_INNER_CANTHUS].y = 0.35 + face_offset_y
    lms[RIGHT_OUTER_CANTHUS].x = 0.70 + face_offset_x
    lms[RIGHT_OUTER_CANTHUS].y = 0.35 + face_offset_y

    # Pálpebras direita
    eyelid_gap_r = 0.005 if right_closed else 0.035
    lms[RIGHT_UPPER_EYELID].x = 0.65 + face_offset_x
    lms[RIGHT_UPPER_EYELID].y = 0.35 - (eyelid_gap_r / 2.0) + face_offset_y
    lms[RIGHT_LOWER_EYELID].x = 0.65 + face_offset_x
    lms[RIGHT_LOWER_EYELID].y = 0.35 + (eyelid_gap_r / 2.0) + face_offset_y

    # Íris direita
    for idx in RIGHT_IRIS_ALL:
        lms[idx].x = 0.65 + face_offset_x
        lms[idx].y = 0.35 + face_offset_y

    # Pontos de pose
    lms[NOSE_TIP].x = 0.50 + face_offset_x
    lms[NOSE_TIP].y = 0.50 + face_offset_y
    lms[CHIN].x = 0.50 + face_offset_x
    lms[CHIN].y = 0.70 + face_offset_y
    lms[MOUTH_LEFT].x = 0.42 + face_offset_x
    lms[MOUTH_LEFT].y = 0.60 + face_offset_y
    lms[MOUTH_RIGHT].x = 0.58 + face_offset_x
    lms[MOUTH_RIGHT].y = 0.60 + face_offset_y

    return lms


class TestGazeFeatures:
    def test_invariance_under_face_translation(self):
        """
        Garante que a posição relativa da íris (rel_x, rel_y) seja invariante
        quando o rosto translada na imagem, enquanto a coordenada absoluta sofre desvio.
        """
        extractor = GazeFeatureExtractor()

        # Rosto na posição base (centro)
        lms_base = create_synthetic_landmarks(face_offset_x=0.0, face_offset_y=0.0)
        feat_base = extractor.extract_features(lms_base)

        # Rosto transladado em 15% na horizontal e 10% na vertical
        lms_moved = create_synthetic_landmarks(face_offset_x=0.15, face_offset_y=0.10)
        feat_moved = extractor.extract_features(lms_moved)

        # As coordenadas absolutas DEVEM mudar pelo valor da translação
        assert np.isclose(feat_moved.fused_abs_gaze[0] - feat_base.fused_abs_gaze[0], 0.15, atol=1e-3)
        assert np.isclose(feat_moved.fused_abs_gaze[1] - feat_base.fused_abs_gaze[1], 0.10, atol=1e-3)

        # A coordenada RELATIVA deve permanecer praticamente IDÊNTICA!
        diff_rel_x = abs(feat_moved.fused_rel_gaze[0] - feat_base.fused_rel_gaze[0])
        diff_rel_y = abs(feat_moved.fused_rel_gaze[1] - feat_base.fused_rel_gaze[1])

        assert diff_rel_x < 1e-4, f"rel_x não é invariante a translação: diff={diff_rel_x}"
        assert diff_rel_y < 1e-4, f"rel_y não é invariante a translação: diff={diff_rel_y}"

    def test_monocular_fallback_when_left_eye_closed(self):
        """Quando o olho esquerdo pisca/fecha, o sistema deve utilizar o olho direito."""
        extractor = GazeFeatureExtractor()
        lms = create_synthetic_landmarks(left_closed=True, right_closed=False)
        feat = extractor.extract_features(lms)

        assert feat.left_eye.is_valid is False
        assert feat.right_eye.is_valid is True
        assert feat.dominant_eye == "right"
        assert feat.tracking_valid is True
        assert np.allclose(feat.fused_rel_gaze, [feat.right_eye.rel_x, feat.right_eye.rel_y])

    def test_monocular_fallback_when_right_eye_closed(self):
        """Quando o olho direito pisca/fecha, o sistema deve utilizar o olho esquerdo."""
        extractor = GazeFeatureExtractor()
        lms = create_synthetic_landmarks(left_closed=False, right_closed=True)
        feat = extractor.extract_features(lms)

        assert feat.left_eye.is_valid is True
        assert feat.right_eye.is_valid is False
        assert feat.dominant_eye == "left"
        assert feat.tracking_valid is True
        assert np.allclose(feat.fused_rel_gaze, [feat.left_eye.rel_x, feat.left_eye.rel_y])

    def test_both_eyes_closed_invalidates_tracking(self):
        """Quando ambos os olhos estão fechados, tracking_valid deve ser False."""
        extractor = GazeFeatureExtractor()
        lms = create_synthetic_landmarks(left_closed=True, right_closed=True)
        feat = extractor.extract_features(lms)

        assert feat.tracking_valid is False
        assert feat.dominant_eye == "none"

    def test_division_by_zero_prevention(self):
        """Coordenadas degeneradas (cantos idênticos) não devem causar ZeroDivisionError."""
        extractor = GazeFeatureExtractor()
        lms = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(500)]
        # Todos os pontos sobrepostos
        feat = extractor.extract_features(lms)
        assert isinstance(feat.fused_rel_gaze, np.ndarray)

    def test_build_feature_vectors(self):
        """Testa geração correta dos vetores para MODEL_A, MODEL_B e MODEL_C."""
        extractor = GazeFeatureExtractor()
        lms = create_synthetic_landmarks()
        feat = extractor.extract_features(lms)

        vec_a = build_feature_vector(feat, "MODEL_A")
        assert len(vec_a) == 6
        assert vec_a[0] == 1.0

        vec_b = build_feature_vector(feat, "MODEL_B")
        assert len(vec_b) == 6
        assert vec_b[0] == 1.0

        vec_c = build_feature_vector(feat, "MODEL_C")
        assert len(vec_c) == 12
        assert vec_c[0] == 1.0
