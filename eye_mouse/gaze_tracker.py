import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os


class GazeTracker:
    def __init__(self, model_filename="face_landmarker.task"):
        # Localizar o arquivo .task
        # Tenta no diretório atual ou no mesmo diretório do script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_filename)

        if not os.path.exists(model_path):
            # Tenta no diretório pai se não achar (caso rodando de fora)
            parent_dir = os.path.dirname(current_dir)
            possible_path = os.path.join(parent_dir, model_filename)
            if os.path.exists(possible_path):
                model_path = possible_path
            else:
                # Fallback para tentar achar em qualquer lugar ou usar o nome direto
                if os.path.exists(model_filename):
                    model_path = model_filename
                else:
                    raise FileNotFoundError(
                        f"Modelo {model_filename} não encontrado em {current_dir}. Baixe-o do site do MediaPipe."
                    )

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
        """Calcula o centro da íris em coordenadas normalizadas (0.0-1.0) relativas à imagem ou absolutas."""

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
        Retorna: (left_iris_center, right_iris_center, landmarks) ou (None, None, None)
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
