import os

# Câmera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# Suavização
EMA_ALPHA = 0.10  # Fator suavização (0.0 = max suave, 1.0 = sem suavização)
DEAD_ZONE_PIXELS = 5  # Movimento mínimo do olho para mover cursor

# Piscada
BLINK_EAR_THRESHOLD = 0.20
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 15
BLINK_COOLDOWN_SEC = 0.5
HOLD_DURATION_SEC = 1.5  # Tempo para ativar modo arrastar

# Calibração
CALIBRATION_POINTS = 9
CALIBRATION_FRAMES_PER_POINT = 30
CALIBRATION_FILE = "calibration_data.npy"

# Tela
SCREEN_MARGIN = 50  # Margem em pixels nas bordas da tela

# Arquivos
LOG_FILE = "eye_mouse.log"
