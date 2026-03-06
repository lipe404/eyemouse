import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from config import get_resource_path, MODEL_FILE


class GazeTracker:
    """
    Rastreia a posição do olhar (íris) usando MediaPipe Face Landmarker.

    Processa frames de vídeo para detectar landmarks faciais e extrair
    a posição das íris.
    """

    def __init__(self, model_filename=MODEL_FILE):
        """
        Inicializa o rastreador de olhar.

        Args:
            model_filename (str): Nome do arquivo de modelo do MediaPipe.

        Raises:
            FileNotFoundError: Se o arquivo de modelo não for encontrado.
        """
        # Localizar o arquivo .task usando helper que suporta PyInstaller
        model_path = get_resource_path(model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Índices da íris (MediaPipe com refine_landmarks=True)
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]

    def get_iris_position(self, landmarks, iris_indices, img_w, img_h):
        """
        Calcula o centro da íris em coordenadas normalizadas (0.0-1.0) relativas à imagem ou absolutas.

        Args:
            landmarks: Lista de landmarks faciais.
            iris_indices (list): Índices dos landmarks da íris.
            img_w (int): Largura da imagem.
            img_h (int): Altura da imagem.

        Returns:
            numpy.ndarray: Coordenadas (x, y) do centro da íris.
        """

        # Coletar pontos da íris
        iris_points = np.array(
            [[landmarks[idx].x, landmarks[idx].y] for idx in iris_indices]
        )

        # Calcular centro (média)
        center = np.mean(iris_points, axis=0)
        return center

    def process_frame(self, frame):
        """
        Processa um frame e retorna as posições das íris.

        Args:
            frame (numpy.ndarray): Frame de vídeo (BGR).

        Returns:
            tuple: (left_iris_center, right_iris_center, landmarks)
                - left_iris_center: Coordenadas da íris esquerda.
                - right_iris_center: Coordenadas da íris direita.
                - landmarks: Objeto com landmarks faciais completos ou None.
        """
        img_h, img_w = frame.shape[:2]

        # Converter BGR para RGB e criar mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detecção
        detection_result = self.detector.detect(mp_image)

        if detection_result.face_landmarks:
            # landmarks é uma lista de objetos NormalizedLandmark
            landmarks = detection_result.face_landmarks[0]

            left_iris = self.get_iris_position(landmarks, self.LEFT_IRIS, img_w, img_h)
            right_iris = self.get_iris_position(
                landmarks, self.RIGHT_IRIS, img_w, img_h
            )

            return left_iris, right_iris, landmarks

        return None, None, None

    def draw_debug(self, frame, landmarks):
        """
        Desenha landmarks da íris e contornos dos olhos para debug visual.

        Args:
            frame (numpy.ndarray): Frame onde desenhar.
            landmarks: Landmarks faciais detectados.
        """
        if not landmarks:
            return

        img_h, img_w = frame.shape[:2]

        # Função auxiliar para converter coordenadas normalizadas -> pixels
        def to_pixel(lm):
            return int(lm.x * img_w), int(lm.y * img_h)

        # Desenhar íris esquerda
        for idx in self.LEFT_IRIS:
            cv2.circle(frame, to_pixel(landmarks[idx]), 1, (0, 255, 0), -1)

        # Desenhar íris direita
        for idx in self.RIGHT_IRIS:
            cv2.circle(frame, to_pixel(landmarks[idx]), 1, (0, 255, 0), -1)

        # Desenhar contorno dos olhos (índices aproximados do MediaPipe)
        # Olho esquerdo: 33, 133, 160, 159, 158, 144, 145, 153
        left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
        pts_left = np.array(
            [to_pixel(landmarks[i]) for i in left_eye_indices], np.int32
        )
        cv2.polylines(frame, [pts_left], True, (255, 255, 0), 1)

        # Olho direito: 362, 263, 387, 386, 385, 373, 374, 380
        right_eye_indices = [362, 263, 387, 386, 385, 373, 374, 380]
        pts_right = np.array(
            [to_pixel(landmarks[i]) for i in right_eye_indices], np.int32
        )
        cv2.polylines(frame, [pts_right], True, (255, 255, 0), 1)
