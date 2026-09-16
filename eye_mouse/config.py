"""
config.py — Configurações centrais do EyeMouse.

Requerimento de Python: >= 3.9 e < 3.13
  MediaPipe não fornece wheels oficiais para Python 3.13+ no Windows.
  Use Python 3.11 (recomendado) ou 3.12.
"""
import os
import sys
import warnings


# ---------------------------------------------------------------------------
# Verificação de versão do Python
# ---------------------------------------------------------------------------
_PY = sys.version_info
if _PY >= (3, 13):
    warnings.warn(
        f"Python {_PY.major}.{_PY.minor} detectado. "
        "O MediaPipe não possui wheels oficiais para Python 3.13+ no Windows. "
        "Use Python 3.11 ou 3.12 para garantir compatibilidade.",
        RuntimeWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Funções de caminho
# ---------------------------------------------------------------------------

def get_resource_path(relative_path):
    """
    Retorna o caminho absoluto para recursos, funcionando para dev e PyInstaller.

    Args:
        relative_path (str): O caminho relativo do recurso.

    Returns:
        str: O caminho absoluto para o recurso.
    """
    try:
        # PyInstaller cria um diretório temporário e armazena o caminho em _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_user_data_dir():
    """
    Retorna o diretório para salvar dados do usuário (calibração, logs).

    Cria o diretório se ele não existir.

    Returns:
        str: O caminho absoluto para o diretório de dados do usuário.
    """
    docs_dir = os.path.join(os.path.expanduser("~"), "Documents", "EyeMouse")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    return docs_dir


# ---------------------------------------------------------------------------
# Diretório de dados
# ---------------------------------------------------------------------------
USER_DATA_DIR = get_user_data_dir()

# ---------------------------------------------------------------------------
# Câmera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# ---------------------------------------------------------------------------
# Suavização do cursor
# ---------------------------------------------------------------------------
EMA_ALPHA = 0.10       # 0.0 = máximo suave, 1.0 = sem suavização
DEAD_ZONE_PIXELS = 5   # Deslocamento mínimo para mover cursor (pixels)

# ---------------------------------------------------------------------------
# Piscada — parâmetros baseados em TEMPO REAL (não em frames)
# Substitui BLINK_MIN_FRAMES / BLINK_MAX_FRAMES para independência de FPS.
# ---------------------------------------------------------------------------
BLINK_EAR_THRESHOLD = 0.20       # Threshold inicial (ajustado dinamicamente)
BLINK_MIN_DURATION_SEC = 0.066   # Duração mínima de uma piscada intencional (~2 fr a 30 FPS)
BLINK_MAX_DURATION_SEC = 0.500   # Duração máxima antes de ser tratado como "hold"
BLINK_COOLDOWN_SEC = 0.5         # Intervalo mínimo entre piscadas consecutivas
HOLD_DURATION_SEC = 1.5          # Tempo de olho fechado para ativar modo arrastar

# Parâmetros legados (mantidos para compatibilidade com testes existentes)
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 15

# Diferenciação de Piscada
BLINK_REFLEX_OPEN_SPEED = 0.08  # EAR/frame — acima disso é reflexo
BLINK_INTENTIONAL_OPEN_SPEED = 0.05

# ---------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------
CALIBRATION_POINTS = 16                          # Grade 4×4
CALIBRATION_FRAMES_PER_POINT = 30
CALIBRATION_REPROJECTION_ERROR_THRESHOLD = 80    # Erro holdout máximo aceitável (px)
CALIBRATION_FILE_PREFIX = os.path.join(USER_DATA_DIR, "calibration_")

# ---------------------------------------------------------------------------
# Tela
# ---------------------------------------------------------------------------
# SCREEN_MARGIN=0 — o cursor alcança toda a área da tela.
# Aumente se precisar de uma zona de segurança nas bordas.
SCREEN_MARGIN = 0

# ---------------------------------------------------------------------------
# Rastreamento
# ---------------------------------------------------------------------------
TRACKING_LOST_TIMEOUT_SEC = 1.0   # Segundos sem rosto para entrar em TRACKING_LOST

# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(USER_DATA_DIR, "eye_mouse.log")
MODEL_FILE = "face_landmarker.task"

# ---------------------------------------------------------------------------
# Feature Flags — defina como False para reverter cada mudança individualmente
# ---------------------------------------------------------------------------
FT_NATIVE_MOUSE    = True   # M1.1: usar OsMouse (SendInput) em vez de pyautogui
FT_SAFETY_RELEASE  = True   # M1.2: release_all() em pausas/erros/encerramento
FT_STATE_MACHINE   = True   # M1.3: máquina de estados AppState
FT_JSON_CALIBRATION = True  # M3.4: persistência JSON (vs .npy com pickle)
FT_HOLDOUT_VALIDATION = True  # M3.2: validação com holdout separado

# ---------------------------------------------------------------------------
# Modo Benchmark
# ---------------------------------------------------------------------------
# Quando True: roda todo o pipeline mas NÃO move o cursor real.
# Use para medir latência de software sem interferir no sistema operacional.
BENCHMARK_MODE = False
