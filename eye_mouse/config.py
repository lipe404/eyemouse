import os
import sys


def get_resource_path(relative_path):
    """Retorna o caminho absoluto para recursos, funcionando para dev e PyInstaller."""
    try:
        # PyInstaller cria um diretório temporário e armazena o caminho em _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_user_data_dir():
    """Retorna o diretório para salvar dados do usuário (calibração, logs)."""
    # Usa a pasta Documentos/EyeMouse
    docs_dir = os.path.join(os.path.expanduser("~"), "Documents", "EyeMouse")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    return docs_dir


# Diretório de dados do usuário
USER_DATA_DIR = get_user_data_dir()

# Câmera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# Suavização
EMA_ALPHA = 0.10  # Fator suavização (0.0 = max suave, 1.0 = sem suavização)
DEAD_ZONE_PIXELS = 5  # Movimento mínimo do olho para mover cursor

# Piscada
BLINK_EAR_THRESHOLD = 0.20 # Valor padrão (será ajustado dinamicamente)
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 15
BLINK_COOLDOWN_SEC = 0.5
HOLD_DURATION_SEC = 1.5  # Tempo para ativar modo arrastar

# Diferenciação de Piscada (Novos parâmetros)
BLINK_REFLEX_OPEN_SPEED = 0.08 # Velocidade de abertura (EAR/frame) acima disso é reflexo
BLINK_INTENTIONAL_OPEN_SPEED = 0.05 # Abaixo disso é intencional

# Calibração
CALIBRATION_POINTS = 9
CALIBRATION_FRAMES_PER_POINT = 30
CALIBRATION_FILE = os.path.join(USER_DATA_DIR, "calibration_data.npy")

# Tela
SCREEN_MARGIN = 50  # Margem em pixels nas bordas da tela

# Arquivos
LOG_FILE = os.path.join(USER_DATA_DIR, "eye_mouse.log")
MODEL_FILE = "face_landmarker.task"  # Nome do arquivo de modelo (será resolvido via get_resource_path)
