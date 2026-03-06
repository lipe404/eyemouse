import numpy as np
import time

class SmoothingFilter:
    def __init__(self, alpha=0.1, dead_zone=5):
        self.alpha = alpha
        self.dead_zone = dead_zone
        self.prev_x = None
        self.prev_y = None
        
    def update(self, x, y):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = x
            self.prev_y = y
            return x, y
            
        # Calcular distância do movimento
        dx = x - self.prev_x
        dy = y - self.prev_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Zona morta
        if dist < self.dead_zone:
            return self.prev_x, self.prev_y
            
        # Aceleração adaptativa (opcional, mas recomendada)
        # Se o movimento for muito grande, aumenta o alpha para responder mais rápido
        current_alpha = self.alpha
        if dist > 100:
            current_alpha = min(1.0, self.alpha * 2.0)
            
        # Filtro EMA
        new_x = current_alpha * x + (1 - current_alpha) * self.prev_x
        new_y = current_alpha * y + (1 - current_alpha) * self.prev_y
        
        self.prev_x = new_x
        self.prev_y = new_y
        
        return new_x, new_y

    def reset(self):
        self.prev_x = None
        self.prev_y = None

    def set_alpha(self, alpha):
        self.alpha = alpha
