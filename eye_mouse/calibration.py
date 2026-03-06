import numpy as np
import os
from config import CALIBRATION_FILE

class CalibrationManager:
    def __init__(self):
        self.iris_points = []
        self.screen_points = []
        self.coeffs_x = None
        self.coeffs_y = None
        self.is_calibrated = False

    def add_point(self, iris_pos, screen_pos):
        """Adiciona um par de pontos (íris, tela) para calibração."""
        self.iris_points.append(iris_pos)
        self.screen_points.append(screen_pos)

    def compute_calibration(self):
        """Calcula os coeficientes de regressão polinomial (2ª ordem)."""
        if len(self.iris_points) < 6:
            print("Pontos insuficientes para calibração (mínimo 6).")
            return False

        iris_data = np.array(self.iris_points)
        screen_data = np.array(self.screen_points)

        # Matriz de design para polinômio de 2ª ordem: [1, x, y, xy, x^2, y^2]
        # x = iris_x, y = iris_y
        X = iris_data[:, 0]
        Y = iris_data[:, 1]
        
        # Construir matriz A
        ones = np.ones(len(X))
        A = np.column_stack([ones, X, Y, X*Y, X**2, Y**2])
        
        # Resolver para Screen X
        # A * coeffs_x = Screen_X
        # Usar lstsq para encontrar a melhor solução (mínimos quadrados)
        self.coeffs_x, _, _, _ = np.linalg.lstsq(A, screen_data[:, 0], rcond=None)
        
        # Resolver para Screen Y
        self.coeffs_y, _, _, _ = np.linalg.lstsq(A, screen_data[:, 1], rcond=None)
        
        self.is_calibrated = True
        self.save_calibration()
        return True

    def map_to_screen(self, iris_pos):
        """Mapeia a posição da íris para coordenadas de tela usando o modelo calibrado."""
        if not self.is_calibrated or self.coeffs_x is None:
            return None
            
        x, y = iris_pos
        # Vetor de features: [1, x, y, xy, x^2, y^2]
        features = np.array([1, x, y, x*y, x**2, y**2])
        
        screen_x = np.dot(features, self.coeffs_x)
        screen_y = np.dot(features, self.coeffs_y)
        
        return int(screen_x), int(screen_y)

    def save_calibration(self):
        """Salva os coeficientes em arquivo .npy."""
        if self.is_calibrated:
            np.save(CALIBRATION_FILE, {
                'coeffs_x': self.coeffs_x,
                'coeffs_y': self.coeffs_y
            })
            print(f"Calibração salva em {CALIBRATION_FILE}")

    def load_calibration(self):
        """Carrega a calibração do arquivo se existir."""
        if os.path.exists(CALIBRATION_FILE):
            try:
                data = np.load(CALIBRATION_FILE, allow_pickle=True).item()
                self.coeffs_x = data['coeffs_x']
                self.coeffs_y = data['coeffs_y']
                self.is_calibrated = True
                print("Calibração carregada com sucesso.")
                return True
            except Exception as e:
                print(f"Erro ao carregar calibração: {e}")
                return False
        return False
