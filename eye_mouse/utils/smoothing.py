import numpy as np
import cv2
import time
from collections import deque


class SmoothingFilter:
    def __init__(self, process_noise=1.0, measurement_noise=1e-1):
        """
        Inicializa o filtro de Kalman para suavização de movimento do mouse.
        
        Estado: [x, y, dx, dy] (posição e velocidade)
        Medição: [x, y] (posição observada)
        """
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Matriz de Medição (H): Observamos apenas x e y
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        
        # Matriz de Transição (F): Será atualizada com dt a cada frame
        # [[1, 0, dt, 0],
        #  [0, 1, 0, dt],
        #  [0, 0, 1,  0],
        #  [0, 0, 0,  1]]
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        # Covariância do Ruído do Processo (Q)
        # Aumentado para 1.0 para permitir movimentos rápidos (alta agilidade)
        self.process_noise_base = process_noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Covariância do Ruído da Medição (R)
        self.base_measurement_noise = measurement_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Covariância do Erro a Posteriori (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.last_time = None
        self.first_update = True
        self.prev_output = None
        
        # Configurações Avançadas
        self.position_buffer = deque(maxlen=5) # Buffer circular para predição
        self.dead_zone_rest = 15.0 # Pixels (repouso)
        self.dead_zone_active = 2.0 # Pixels (movimento rápido)
        
        # Fator de suavização base (pode ser ajustado pelo usuário via set_alpha)
        self.current_alpha = 0.5 

    def _sigmoid(self, x, midpoint=50, steepness=0.1):
        """Curva sigmóide para transição suave baseada na velocidade."""
        return 1 / (1 + np.exp(-steepness * (x - midpoint)))

    def update(self, x, y):
        current_time = time.time()
        
        # Adicionar ao buffer
        self.position_buffer.append((x, y))
        
        # Vetor de medição
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        # Inicialização no primeiro frame
        if self.first_update:
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.last_time = current_time
            self.first_update = False
            self.prev_output = (x, y)
            return x, y

        # Calcular delta tempo (dt)
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt <= 0: dt = 1.0/30.0
        elif dt > 1.0: dt = 1.0/30.0

        # Atualizar matriz de transição com dt real
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        # --- 2. Suavização Separada por Eixo com Velocidade & 4. Aceleração Adaptativa ---
        # Obter velocidade estimada atual (do estado anterior ou medição bruta)
        # Usaremos a velocidade do estado anterior do Kalman para estabilidade
        vx_est = abs(self.kf.statePost[2][0])
        vy_est = abs(self.kf.statePost[3][0])
        
        # Calcular fator de resposta (0.0 a 1.0) usando sigmóide
        # Velocidade baixa -> response baixo (suave)
        # Velocidade alta -> response alto (rápido)
        # Midpoint=100px/s, Steepness=0.05
        response_x = self._sigmoid(vx_est, midpoint=100, steepness=0.05)
        response_y = self._sigmoid(vy_est, midpoint=100, steepness=0.05)
        
        # Ajustar R (Measurement Noise) independentemente para X e Y
        # Baseado no alpha definido pelo usuário, modulado pela velocidade
        # Se usuário definiu alpha alto (rápido), mantemos rápido.
        # Se alpha baixo (suave), aceleramos apenas se necessário.
        
        base_r = self.base_measurement_noise
        
        # Mapeamento: Response alto -> R baixo (confia na medição)
        # Response baixo -> R alto (confia no modelo/suaviza)
        # Fator de modulação: 
        # r_dynamic = base_r / (1 + factor * response)
        
        # Vamos usar uma abordagem direta na matriz R diagonal
        r_x = base_r / (1.0 + 10.0 * response_x) 
        r_y = base_r / (1.0 + 10.0 * response_y)
        
        self.kf.measurementNoiseCov[0, 0] = r_x
        self.kf.measurementNoiseCov[1, 1] = r_y

        # 1. Predição
        prediction = self.kf.predict()
        
        # 2. Correção
        estimated = self.kf.correct(measurement)
        
        est_x = estimated[0][0]
        est_y = estimated[1][0]
        est_vx = estimated[2][0]
        est_vy = estimated[3][0]

        # --- 3. Dead Zone Dinâmica ---
        vel_magnitude = np.sqrt(est_vx**2 + est_vy**2)
        
        # Interpolar tamanho da dead zone baseada na velocidade
        # Vel 0 -> 15px, Vel 500 -> 2px
        dz_factor = max(0, min(1, vel_magnitude / 500.0))
        current_dead_zone = self.dead_zone_rest * (1 - dz_factor) + self.dead_zone_active * dz_factor
        
        if self.prev_output:
            prev_x, prev_y = self.prev_output
            dist = np.sqrt((est_x - prev_x)**2 + (est_y - prev_y)**2)
            
            if dist < current_dead_zone:
                est_x, est_y = prev_x, prev_y
                est_vx, est_vy = 0, 0 # Zerar velocidade efetiva se estiver na dead zone

        # --- 5. Predição de Movimento (Buffer Circular) ---
        # Usar a velocidade para projetar o cursor levemente à frente para compensar lag
        # Lookahead de ~2 frames (66ms)
        lookahead = 0.066 
        
        final_x = est_x + est_vx * lookahead
        final_y = est_y + est_vy * lookahead
        
        self.prev_output = (final_x, final_y)
        
        return final_x, final_y

    def reset(self):
        self.first_update = True
        self.last_time = None
        self.kf.statePost = np.zeros((4, 1), np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.position_buffer.clear()

    def set_alpha(self, alpha):
        """
        Define o nível base de suavização.
        Alpha 1.0 = Rápido (R baixo)
        Alpha 0.0 = Lento (R alto)
        """
        # Clampar alpha
        alpha = max(0.01, min(1.0, alpha))
        self.current_alpha = alpha
        
        # Recalcular R base
        exponent = (1.0 - alpha) * 4.0 - 3.0
        self.base_measurement_noise = 10.0 ** exponent
