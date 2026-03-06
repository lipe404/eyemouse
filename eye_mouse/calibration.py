import numpy as np
import os
from config import CALIBRATION_FILE_PREFIX


class CalibrationManager:
    def __init__(self, profile_name="default"):
        self.profile_name = profile_name
        self.calibration_file = f"{CALIBRATION_FILE_PREFIX}{profile_name}.npy"
        self.iris_points = []
        self.screen_points = []
        self.coeffs_x = None
        self.coeffs_y = None
        self.is_calibrated = False

    def set_profile(self, profile_name):
        self.profile_name = profile_name
        self.calibration_file = f"{CALIBRATION_FILE_PREFIX}{profile_name}.npy"
        # Tentar carregar a calibração do novo perfil
        self.load_calibration()

    def add_point(self, iris_pos, screen_pos):
        """Adiciona um par de pontos (íris, tela) para calibração."""
        self.iris_points.append(iris_pos)
        self.screen_points.append(screen_pos)

    def clear_points(self):
        self.iris_points = []
        self.screen_points = []

    def compute_calibration(self):
        """Calcula os coeficientes de regressão polinomial (2ª ordem) e valida."""
        if len(self.iris_points) < 6:
            print("Pontos insuficientes para calibração (mínimo 6).")
            return False, 0.0

        iris_data = np.array(self.iris_points)
        screen_data = np.array(self.screen_points)

        # Matriz de design para polinômio de 2ª ordem: [1, x, y, xy, x^2, y^2]
        # x = iris_x, y = iris_y
        X = iris_data[:, 0]
        Y = iris_data[:, 1]

        # Construir matriz A
        ones = np.ones(len(X))
        A = np.column_stack([ones, X, Y, X * Y, X**2, Y**2])

        # Resolver para Screen X
        # A * coeffs_x = Screen_X
        # Usar lstsq para encontrar a melhor solução (mínimos quadrados)
        self.coeffs_x, _, _, _ = np.linalg.lstsq(A, screen_data[:, 0], rcond=None)

        # Resolver para Screen Y
        self.coeffs_y, _, _, _ = np.linalg.lstsq(A, screen_data[:, 1], rcond=None)

        self.is_calibrated = True
        
        # Validação (Melhoria 12)
        mean_error = self._validate_calibration()
        print(f"Erro médio de reprojeção: {mean_error:.2f}px")

        self.save_calibration()
        return True, mean_error

    def _validate_calibration(self):
        """Calcula o erro médio de reprojeção."""
        if not self.is_calibrated:
            return float('inf')
            
        total_error = 0
        count = 0
        
        for i, iris_pt in enumerate(self.iris_points):
            screen_pt = self.screen_points[i]
            predicted = self.map_to_screen(iris_pt)
            if predicted:
                dist = np.linalg.norm(np.array(predicted) - np.array(screen_pt))
                total_error += dist
                count += 1
        
        return total_error / count if count > 0 else float('inf')

    def map_to_screen(self, iris_pos):
        """Mapeia a posição da íris para coordenadas de tela usando o modelo calibrado."""
        if not self.is_calibrated or self.coeffs_x is None:
            return None

        x, y = iris_pos
        # Vetor de features: [1, x, y, xy, x^2, y^2]
        features = np.array([1, x, y, x * y, x**2, y**2])

        screen_x = np.dot(features, self.coeffs_x)
        screen_y = np.dot(features, self.coeffs_y)

        return int(screen_x), int(screen_y)

    def save_calibration(self):
        """Salva os coeficientes em arquivo .npy."""
        if self.is_calibrated:
            np.save(
                self.calibration_file, {"coeffs_x": self.coeffs_x, "coeffs_y": self.coeffs_y}
            )
            print(f"Calibração salva em {self.calibration_file}")

    def load_calibration(self):
        """Carrega a calibração do arquivo se existir."""
        if os.path.exists(self.calibration_file):
            try:
                data = np.load(self.calibration_file, allow_pickle=True).item()
                self.coeffs_x = data["coeffs_x"]
                self.coeffs_y = data["coeffs_y"]
                self.is_calibrated = True
                print(f"Calibração carregada de {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Erro ao carregar calibração: {e}")
                return False
        return False
