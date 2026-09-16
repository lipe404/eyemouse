"""
conftest.py — Configuração global do pytest.

Adiciona eye_mouse/ ao sys.path e injeta stubs de módulos pesados
(mediapipe, pyautogui, cv2, keyboard) para que todos os testes possam
importar os módulos da aplicação sem ter essas bibliotecas instaladas.
"""
import sys
import os
from unittest.mock import MagicMock

# Add the eye_mouse directory to sys.path so that 'import config' works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'eye_mouse')))

# ---------------------------------------------------------------------------
# Stubs de módulos que não estão instalados no ambiente de teste
# (mediapipe, pyautogui, cv2, keyboard)
# ---------------------------------------------------------------------------
import types

# mediapipe hierarchy
try:
    import mediapipe
except ImportError:
    mp_mod = types.ModuleType('mediapipe')
    mp_mod.__path__ = []
    tasks_mod = types.ModuleType('mediapipe.tasks')
    tasks_mod.__path__ = []
    py_mod = types.ModuleType('mediapipe.tasks.python')
    py_mod.__path__ = []
    vis_mod = MagicMock()

    mp_mod.tasks = tasks_mod
    tasks_mod.python = py_mod
    py_mod.vision = vis_mod
    mp_mod.Image = MagicMock()
    mp_mod.ImageFormat = MagicMock()

    sys.modules['mediapipe'] = mp_mod
    sys.modules['mediapipe.tasks'] = tasks_mod
    sys.modules['mediapipe.tasks.python'] = py_mod
    sys.modules['mediapipe.tasks.python.vision'] = vis_mod

# pyautogui
try:
    import pyautogui
except ImportError:
    _pag_stub = MagicMock()
    _pag_stub.size.return_value = (1920, 1080)
    _pag_stub.PAUSE = 0.1
    _pag_stub.FAILSAFE = True
    sys.modules['pyautogui'] = _pag_stub

# cv2
try:
    import cv2
except ImportError:
    _cv2_stub = MagicMock()
    _cv2_stub.VideoCapture = MagicMock()
    _cv2_stub.CAP_PROP_FRAME_WIDTH = 3
    _cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
    _cv2_stub.resize = MagicMock(return_value=MagicMock())
    _cv2_stub.cvtColor = MagicMock()
    sys.modules['cv2'] = _cv2_stub

# keyboard
try:
    import keyboard
except ImportError:
    _kbd_stub = MagicMock()
    sys.modules['keyboard'] = _kbd_stub
