"""
gaze_features.py — Extração de features oculares relativas, pose da cabeça e fusão de olhos.

Milestone 3:
  1. Extração de posição da íris normalizada em relação à geometria de cada olho
     (canthus interno e externo, eixo ocular e abertura palpebral).
  2. Compensação de rotação facial e estimativa de pose 3D (Yaw, Pitch, Roll).
  3. Fusão ponderada e seleção monocular inteligente com indicadores de qualidade independentes.
  4. Três modelos intercambiáveis:
       - Model A: Coordenadas absolutas originais.
       - Model B: Posição relativa das íris (invariante a translação facial pura).
       - Model C: Posição relativa + compensação de pose de cabeça (Yaw, Pitch, Roll, escala).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Epsilon numérico para prevenir divisões por zero
EPS = 1e-6

# Índices canônicos dos landmarks faciais do MediaPipe FaceMesh
# Olho Esquerdo (do observador ou sujeito)
LEFT_INNER_CANTHUS = 133
LEFT_OUTER_CANTHUS = 33
LEFT_UPPER_EYELID = 159
LEFT_LOWER_EYELID = 145
LEFT_IRIS_CENTER = 468
LEFT_IRIS_ALL = [468, 469, 470, 471, 472]

# Olho Direito
RIGHT_INNER_CANTHUS = 362
RIGHT_OUTER_CANTHUS = 263
RIGHT_UPPER_EYELID = 386
RIGHT_LOWER_EYELID = 374
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_ALL = [473, 474, 475, 476, 477]

# Pontos de referência para Pose da Cabeça (solvePnP)
NOSE_TIP = 1
CHIN = 152
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Modelo 3D genérico de face para solvePnP (coordenadas antropométricas médias em mm)
FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],          # Ponta do nariz
    [0.0, -330.0, -65.0],     # Queixo
    [-225.0, 170.0, -135.0],  # Canto externo olho esquerdo
    [225.0, 170.0, -135.0],   # Canto externo olho direito
    [-150.0, -150.0, -125.0], # Canto esquerdo da boca
    [150.0, -150.0, -125.0],  # Canto direito da boca
], dtype=np.float64)


@dataclass(frozen=True)
class EyeMetrics:
    """Métricas geométricas e de qualidade para um olho individual."""
    iris_center: np.ndarray          # (x, y) normalizado da íris
    inner_canthus: np.ndarray        # (x, y) canto interno
    outer_canthus: np.ndarray        # (x, y) canto externo
    eye_width: float                 # Largura em coords normalizadas
    eye_height: float                # Altura (abertura) em coords normalizadas
    ear: float                       # Eye Aspect Ratio (altura / largura)
    rel_x: float                     # Posição horizontal relativa [0..1]
    rel_y: float                     # Posição vertical relativa
    quality: float                   # Qualidade do rastreamento (0.0 a 1.0)
    is_valid: bool                   # True se olho aberto e íris rastreada


@dataclass(frozen=True)
class HeadPose:
    """Orientação e posição da cabeça."""
    yaw: float                       # Rotação horizontal (graus: negativo=esq, positivo=dir)
    pitch: float                     # Rotação vertical (graus: negativo=baixo, positivo=cima)
    roll: float                      # Inclinação lateral (graus: negativo=anti-horário, positivo=horário)
    face_center: np.ndarray          # (x, y) centro facial aproximado
    face_scale: float                # Escala / distância proxy (largura dos olhos)
    is_valid: bool = True


@dataclass(frozen=True)
class CombinedGazeFeatures:
    """Vetor completo de features combinadas para calibração e predição."""
    left_eye: EyeMetrics
    right_eye: EyeMetrics
    head_pose: HeadPose
    fused_rel_gaze: np.ndarray       # (rel_x, rel_y) combinado ponderado
    fused_abs_gaze: np.ndarray       # (abs_x, abs_y) média absoluta clássica
    dominant_eye: str                # "both", "left", "right", ou "none"
    tracking_valid: bool             # True se ao menos um olho for válido
    quality_score: float             # Qualidade global combinada


class GazeFeatureExtractor:
    """
    Extrator de features geométricas avançadas do olhar e pose da cabeça.
    """

    def __init__(self, ear_closed_threshold: float = 0.15):
        self.ear_closed_threshold = ear_closed_threshold

    def extract_eye_metrics(
        self,
        landmarks: Any,
        iris_center_idx: int,
        iris_indices: List[int],
        inner_idx: int,
        outer_idx: int,
        upper_idx: int,
        lower_idx: int,
        is_left: bool = True,
    ) -> EyeMetrics:
        """Calcula a geometria e posição relativa da íris para um olho."""
        def get_pt(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x, landmarks[idx].y], dtype=np.float64)

        iris_center = np.mean([get_pt(i) for i in iris_indices], axis=0)
        inner = get_pt(inner_idx)
        outer = get_pt(outer_idx)
        upper = get_pt(upper_idx)
        lower = get_pt(lower_idx)

        # Vetor do eixo do olho (do canto externo para o canto interno)
        eye_axis = inner - outer
        eye_width = float(np.linalg.norm(eye_axis)) + EPS
        axis_u = eye_axis / eye_width

        # Vetor perpendicular ao eixo ocular (ortogonal 2D para cima)
        axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)

        # Altura do olho (distância entre pálpebras ao longo de axis_v)
        lid_vector = upper - lower
        eye_height = float(abs(np.dot(lid_vector, axis_v))) + EPS
        ear = float(eye_height / eye_width)

        # Projeção da íris relativa ao canto externo e centro ocular
        eye_center = (inner + outer) / 2.0
        iris_from_outer = iris_center - outer
        iris_from_center = iris_center - eye_center

        rel_x = float(np.dot(iris_from_outer, axis_u) / eye_width)
        rel_y = float(np.dot(iris_from_center, axis_v) / (eye_height + EPS))

        # Avaliação de qualidade do olho
        is_open = ear >= self.ear_closed_threshold
        # Posição da íris deve estar razoavelmente contida nas proximidades da fenda palpebral
        in_bounds = (-0.2 <= rel_x <= 1.2) and (-1.5 <= rel_y <= 1.5)

        if not is_open:
            quality = 0.0
            is_valid = False
        elif not in_bounds:
            quality = 0.2
            is_valid = False
        else:
            # Score normalizado baseado na abertura e consistência
            quality = min(1.0, max(0.0, (ear - self.ear_closed_threshold) / (0.15 + EPS)))
            is_valid = quality > 0.2

        return EyeMetrics(
            iris_center=iris_center,
            inner_canthus=inner,
            outer_canthus=outer,
            eye_width=eye_width,
            eye_height=eye_height,
            ear=ear,
            rel_x=rel_x,
            rel_y=rel_y,
            quality=quality,
            is_valid=is_valid,
        )

    def estimate_head_pose(
        self,
        landmarks: Any,
        img_w: int = 640,
        img_h: int = 480,
    ) -> HeadPose:
        """
        Estima a orientação 3D da cabeça (Yaw, Pitch, Roll) e posição central.
        Usa solvePnP com fallback para geometria 2D estável.
        """
        def pt2d(idx: int) -> Tuple[float, float]:
            return landmarks[idx].x * img_w, landmarks[idx].y * img_h

        left_eye_outer = np.array([landmarks[LEFT_OUTER_CANTHUS].x, landmarks[LEFT_OUTER_CANTHUS].y])
        right_eye_outer = np.array([landmarks[RIGHT_OUTER_CANTHUS].x, landmarks[RIGHT_OUTER_CANTHUS].y])
        eye_span = right_eye_outer - left_eye_outer
        face_scale = float(np.linalg.norm(eye_span)) + EPS

        # Centro da face aproximado
        face_center = (left_eye_outer + right_eye_outer) / 2.0

        # Roll 2D (inclinação no plano da imagem em graus)
        roll_deg = math.degrees(math.atan2(eye_span[1], eye_span[0]))

        # Tentar solvePnP completo se possível
        image_points = np.array([
            pt2d(NOSE_TIP),
            pt2d(CHIN),
            pt2d(LEFT_OUTER_CANTHUS),
            pt2d(RIGHT_OUTER_CANTHUS),
            pt2d(MOUTH_LEFT),
            pt2d(MOUTH_RIGHT),
        ], dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2.0],
            [0, focal_length, img_h / 2.0],
            [0, 0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        yaw_deg, pitch_deg = 0.0, 0.0
        success, rvec, tvec = False, None, None

        try:
            success, rvec, tvec = cv2.solvePnP(
                FACE_MODEL_3D, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception:
            success = False

        if success and rvec is not None:
            rmat, _ = cv2.Rodrigues(rvec)
            # Decomposição em ângulos de Euler
            sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
            singular = sy < 1e-6

            if not singular:
                pitch_rad = math.atan2(rmat[2, 1], rmat[2, 2])
                yaw_rad = math.atan2(-rmat[2, 0], sy)
                roll_rad = math.atan2(rmat[1, 0], rmat[0, 0])
            else:
                pitch_rad = math.atan2(-rmat[1, 2], rmat[1, 1])
                yaw_rad = math.atan2(-rmat[2, 0], sy)
                roll_rad = 0.0

            pitch_deg = math.degrees(pitch_rad)
            yaw_deg = math.degrees(yaw_rad)
            roll_deg = math.degrees(roll_rad)
        else:
            # Fallback geométrico aproximado para Yaw e Pitch
            nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
            # Assimetria do nariz em relação aos cantos dos olhos
            d_left = np.linalg.norm(nose - left_eye_outer)
            d_right = np.linalg.norm(nose - right_eye_outer)
            yaw_deg = float((d_right - d_left) / (face_scale + EPS) * 60.0)

            # Pitch baseado na distância vertical nariz-olhos
            vertical_disp = nose[1] - face_center[1]
            pitch_deg = float((vertical_disp / (face_scale + EPS) - 0.5) * 60.0)

        return HeadPose(
            yaw=yaw_deg,
            pitch=pitch_deg,
            roll=roll_deg,
            face_center=face_center,
            face_scale=face_scale,
            is_valid=True,
        )

    def extract_features(
        self,
        landmarks: Any,
        img_w: int = 640,
        img_h: int = 480,
    ) -> CombinedGazeFeatures:
        """
        Extrai conjunto completo de features com fusão inteligente dos olhos.
        """
        left = self.extract_eye_metrics(
            landmarks=landmarks,
            iris_center_idx=LEFT_IRIS_CENTER,
            iris_indices=LEFT_IRIS_ALL,
            inner_idx=LEFT_INNER_CANTHUS,
            outer_idx=LEFT_OUTER_CANTHUS,
            upper_idx=LEFT_UPPER_EYELID,
            lower_idx=LEFT_LOWER_EYELID,
            is_left=True,
        )

        right = self.extract_eye_metrics(
            landmarks=landmarks,
            iris_center_idx=RIGHT_IRIS_CENTER,
            iris_indices=RIGHT_IRIS_ALL,
            inner_idx=RIGHT_INNER_CANTHUS,
            outer_idx=RIGHT_OUTER_CANTHUS,
            upper_idx=RIGHT_UPPER_EYELID,
            lower_idx=RIGHT_LOWER_EYELID,
            is_left=False,
        )

        head_pose = self.estimate_head_pose(landmarks, img_w, img_h)

        # Média clássica absoluta (Modelo A)
        fused_abs = (left.iris_center + right.iris_center) / 2.0

        # Fusão ponderada das posições relativas (Modelos B e C)
        total_quality = left.quality + right.quality
        if left.is_valid and right.is_valid and total_quality > EPS:
            w_left = left.quality / total_quality
            w_right = right.quality / total_quality
            fused_rel = np.array([
                w_left * left.rel_x + w_right * right.rel_x,
                w_left * left.rel_y + w_right * right.rel_y,
            ], dtype=np.float64)
            dominant = "both"
            tracking_valid = True
            quality_score = float(total_quality / 2.0)
        elif left.is_valid:
            fused_rel = np.array([left.rel_x, left.rel_y], dtype=np.float64)
            dominant = "left"
            tracking_valid = True
            quality_score = left.quality
        elif right.is_valid:
            fused_rel = np.array([right.rel_x, right.rel_y], dtype=np.float64)
            dominant = "right"
            tracking_valid = True
            quality_score = right.quality
        else:
            fused_rel = np.array([0.5, 0.0], dtype=np.float64)
            dominant = "none"
            tracking_valid = False
            quality_score = 0.0

        return CombinedGazeFeatures(
            left_eye=left,
            right_eye=right,
            head_pose=head_pose,
            fused_rel_gaze=fused_rel,
            fused_abs_gaze=fused_abs,
            dominant_eye=dominant,
            tracking_valid=tracking_valid,
            quality_score=quality_score,
        )


def build_feature_vector(features: CombinedGazeFeatures, model_type: str = "MODEL_C") -> np.ndarray:
    """
    Constrói o vetor de entrada X para o regressor de calibração conforme o modelo selecionado.

    Modelos:
      - MODEL_A: [1, x_abs, y_abs, x_abs*y_abs, x_abs^2, y_abs^2] (método clássico).
      - MODEL_B: [1, rel_x, rel_y, rel_x*rel_y, rel_x^2, rel_y^2] (íris relativa).
      - MODEL_C: [1, rel_x, rel_y, yaw/100, pitch/100, roll/100, face_x, face_y, scale] (com pose).
    """
    model_type = model_type.upper()

    if model_type == "MODEL_A":
        ax, ay = features.fused_abs_gaze[0], features.fused_abs_gaze[1]
        return np.array([1.0, ax, ay, ax * ay, ax**2, ay**2], dtype=np.float64)

    elif model_type == "MODEL_B":
        rx, ry = features.fused_rel_gaze[0], features.fused_rel_gaze[1]
        return np.array([1.0, rx, ry, rx * ry, rx**2, ry**2], dtype=np.float64)

    elif model_type == "MODEL_C":
        rx, ry = features.fused_rel_gaze[0], features.fused_rel_gaze[1]
        yaw_norm = features.head_pose.yaw / 50.0
        pitch_norm = features.head_pose.pitch / 50.0
        roll_norm = features.head_pose.roll / 50.0
        fx, fy = features.head_pose.face_center[0], features.head_pose.face_center[1]
        scale = features.head_pose.face_scale

        return np.array([
            1.0, rx, ry, rx * ry, rx**2, ry**2,
            yaw_norm, pitch_norm, roll_norm,
            fx, fy, scale
        ], dtype=np.float64)

    else:
        raise ValueError(f"Modelo desconhecido: {model_type}. Opções válidas: MODEL_A, MODEL_B, MODEL_C")
