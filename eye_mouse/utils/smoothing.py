import numpy as np
import cv2
import time


class SmoothingFilter:
    def __init__(self, process_noise=1.0, measurement_noise=1e-1):
        """
        Inicializa o filtro de Kalman para suavização de movimento do mouse.

        Estado: [x, y, dx, dy] (posição e velocidade)
        Medição: [x, y] (posição observada)
        """
        self.kf = cv2.KalmanFilter(4, 2)

        # Matriz de Medição (H): Observamos apenas x e y
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)

        # Matriz de Transição (F): Será atualizada com dt a cada frame
        # [[1, 0, dt, 0],
        #  [0, 1, 0, dt],
        #  [0, 0, 1,  0],
        #  [0, 0, 0,  1]]
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        # Covariância do Ruído do Processo (Q)
        # Aumentado para 1.0 para permitir movimentos rápidos (alta agilidade)
        self.process_noise_base = process_noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Covariância do Ruído da Medição (R)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Covariância do Erro a Posteriori (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.last_time = None
        self.first_update = True
        self.prev_output = None

    def update(self, x, y):
        current_time = time.time()

        # Vetor de medição
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        # Inicialização no primeiro frame
        if self.first_update:
            self.kf.statePost = np.array(
                [[np.float32(x)], [np.float32(y)], [0], [0]], np.float32
            )
            self.last_time = current_time
            self.first_update = False
            self.prev_output = (x, y)
            return x, y

        # Calcular delta tempo (dt)
        dt = current_time - self.last_time
        self.last_time = current_time

        # Evitar dt zero ou muito grande em casos extremos
        if dt <= 0:
            dt = 1.0 / 30.0
        elif dt > 1.0:
            dt = 1.0 / 30.0  # Resetar se houver um lag muito grande

        # Atualizar matriz de transição com dt real
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        # 1. Predição (baseada no estado anterior e velocidade)
        prediction = self.kf.predict()

        # 2. Correção (atualiza com a nova medição)
        estimated = self.kf.correct(measurement)

        # Extrair posição estimada
        est_x = estimated[0][0]
        est_y = estimated[1][0]

        self.prev_output = (est_x, est_y)

        return est_x, est_y

    def reset(self):
        self.first_update = True
        self.last_time = None
        self.kf.statePost = np.zeros((4, 1), np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def set_alpha(self, alpha):
        """
        Ajusta a suavização dinamicamente.
        Mantemos a assinatura 'set_alpha' para compatibilidade com o código existente.

        alpha (0.0 a 1.0):
          - 1.0: Máxima resposta (menor suavização) -> R baixo
          - 0.0: Máxima suavização (menor resposta) -> R alto
        """
        # Clampar alpha
        alpha = max(0.01, min(1.0, alpha))

        # Mapear alpha para Measurement Noise (R) de forma exponencial
        # alpha=1.0 -> R ~ 0.001 (segue o input)
        # alpha=0.5 -> R ~ 0.1
        # alpha=0.1 -> R ~ 4.0 (suaviza muito)

        # Fórmula: R = 10 ^ ((1-alpha)*4 - 3)
        # Ex: alpha=1.0 -> 10^-3 = 0.001
        # Ex: alpha=0.0 -> 10^1 = 10

        exponent = (1.0 - alpha) * 4.0 - 3.0
        new_r = 10.0**exponent

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * new_r
