import cv2
import mediapipe as mp
import numpy as np

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Índices da íris (MediaPipe com refine_landmarks=True)
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]

    def get_iris_position(self, landmarks, iris_indices, img_w, img_h):
        """Calcula o centro da íris em coordenadas normalizadas (0.0-1.0) relativas à imagem ou absolutas."""
        # Aqui vamos retornar coordenadas normalizadas relativas ao frame da câmera
        # para serem independentes da resolução
        
        # Coletar pontos da íris
        iris_points = np.array([
            [landmarks[idx].x, landmarks[idx].y] for idx in iris_indices
        ])
        
        # Calcular centro (média)
        center = np.mean(iris_points, axis=0)
        return center

    def process_frame(self, frame):
        """
        Processa um frame e retorna as posições das íris.
        Retorna: (left_iris_center, right_iris_center, landmarks) ou (None, None, None)
        """
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_iris = self.get_iris_position(landmarks, self.LEFT_IRIS, img_w, img_h)
            right_iris = self.get_iris_position(landmarks, self.RIGHT_IRIS, img_w, img_h)
            
            return left_iris, right_iris, landmarks
            
        return None, None, None
