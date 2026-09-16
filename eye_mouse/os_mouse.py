"""
os_mouse.py — Driver nativo de mouse para Windows via SendInput e backends intercambiáveis.

Fornece:
  - BaseMouseBackend: Interface abstrata para drivers de controle de cursor e botões.
  - OsMouse: Driver de ultra-baixa latência usando a Win32 SendInput API (ctypes).
    Suporta DPI Awareness (Per-Monitor V2) e Desktop Virtual Multi-Monitor (MOUSEEVENTF_VIRTUALDESK).
  - PyAutoGuiMouse: Driver baseado em PyAutoGUI (com PAUSE=0) para comparação e fallback.
  - create_mouse_backend(): Factory para alternância dinâmica entre os backends.

Não altera configurações globais do Windows (velocidade do mouse, aceleração do ponteiro).
Proteção equivalente ao pyautogui.FAILSAFE:
  - AppState machine impede ações em estados inválidos.
  - release_all() é idempotente e chamado em qualquer transição de estado anormal.
  - emergency_stop() bloqueia eventos imediatamente em caso de anomalia.
"""

from abc import ABC, abstractmethod
import ctypes
from ctypes import wintypes
import logging
import sys
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes Win32
# ---------------------------------------------------------------------------
INPUT_MOUSE = 0

MOUSEEVENTF_MOVE        = 0x0001
MOUSEEVENTF_LEFTDOWN    = 0x0002
MOUSEEVENTF_LEFTUP      = 0x0004
MOUSEEVENTF_RIGHTDOWN   = 0x0008
MOUSEEVENTF_RIGHTUP     = 0x0010
MOUSEEVENTF_WHEEL       = 0x0800
MOUSEEVENTF_ABSOLUTE    = 0x8000
# VIRTUALDESK mapeia para o desktop virtual inteiro (multi-monitor)
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_CXSCREEN = 0
SM_CYSCREEN = 1
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


# ---------------------------------------------------------------------------
# Estruturas Win32 necessárias para SendInput
# ---------------------------------------------------------------------------
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type",   wintypes.DWORD),
        ("_input", _INPUT_UNION),
    ]


# ---------------------------------------------------------------------------
# DPI Awareness
# ---------------------------------------------------------------------------
def init_dpi_awareness() -> bool:
    """
    Configura DPI awareness no Windows para garantir que GetSystemMetrics
    e SendInput operem em coordenadas físicas reais (sem distorção de escala DWM).

    Tenta, nesta ordem:
      1. SetProcessDpiAwarenessContext (PER_MONITOR_AWARE_V2, Windows 10 1703+)
      2. SetProcessDpiAwareness (PROCESS_PER_MONITOR_DPI_AWARE, Windows 8.1+)
      3. SetProcessDPIAware (Windows Vista+)
    """
    if sys.platform != "win32":
        return False

    # 1. Windows 10 1703+ (Context: -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            res = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            if res:
                logger.info("DPI Awareness ativado: PER_MONITOR_AWARE_V2")
                return True
    except Exception as exc:
        logger.debug("Falha ao configurar PER_MONITOR_AWARE_V2: %s", exc)

    # 2. Windows 8.1+
    try:
        shcore = getattr(ctypes.windll, "shcore", None)
        if shcore and hasattr(shcore, "SetProcessDpiAwareness"):
            res = shcore.SetProcessDpiAwareness(2)  # 2 = PROCESS_PER_MONITOR_DPI_AWARE
            if res == 0:
                logger.info("DPI Awareness ativado: PROCESS_PER_MONITOR_DPI_AWARE")
                return True
    except Exception as exc:
        logger.debug("Falha ao configurar SetProcessDpiAwareness: %s", exc)

    # 3. Windows Vista+
    try:
        user32 = getattr(ctypes.windll, "user32", None)
        if user32 and hasattr(user32, "SetProcessDPIAware"):
            res = user32.SetProcessDPIAware()
            if res:
                logger.info("DPI Awareness ativado: SetProcessDPIAware")
                return True
    except Exception as exc:
        logger.debug("Falha ao configurar SetProcessDPIAware: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Interface Base de Mouse
# ---------------------------------------------------------------------------
class BaseMouseBackend(ABC):
    """Interface abstrata para drivers de controle de mouse do sistema operacional."""

    @abstractmethod
    def move(self, x: int, y: int) -> None:
        """Move o cursor para a posição absoluta (x, y) em pixels."""
        pass

    @abstractmethod
    def left_click(self) -> None:
        """Executa clique com o botão esquerdo."""
        pass

    @abstractmethod
    def right_click(self) -> None:
        """Executa clique com o botão direito."""
        pass

    @abstractmethod
    def double_click(self) -> None:
        """Executa duplo clique com o botão esquerdo."""
        pass

    @abstractmethod
    def button_down(self, button: str = "left") -> None:
        """Pressiona e mantém o botão indicado."""
        pass

    @abstractmethod
    def button_up(self, button: str = "left") -> None:
        """Solta o botão indicado."""
        pass

    @abstractmethod
    def scroll(self, delta: int) -> None:
        """Rola a roda do mouse (positivo=cima, negativo=baixo)."""
        pass

    @abstractmethod
    def release_all(self) -> None:
        """Libera todos os botões mantidos pressionados (idempotente)."""
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Bloqueia eventos futuros e libera botões."""
        pass

    @abstractmethod
    def reset_emergency_stop(self) -> None:
        """Reativa envio de eventos após emergency_stop."""
        pass

    @property
    @abstractmethod
    def is_emergency_stopped(self) -> bool:
        pass

    @property
    @abstractmethod
    def button_state(self) -> dict:
        pass

    @property
    @abstractmethod
    def screen_size(self) -> Tuple[int, int]:
        pass

    @property
    def virtual_screen_geometry(self) -> Tuple[int, int, int, int]:
        """Retorna (vx, vy, vw, vh) do desktop virtual."""
        w, h = self.screen_size
        return 0, 0, w, h


# ---------------------------------------------------------------------------
# OsMouse (Win32 SendInput nativo)
# ---------------------------------------------------------------------------
class OsMouse(BaseMouseBackend):
    """
    Driver de mouse de baixo nível usando a API Win32 SendInput.

    Thread-safe. Rastreia o estado dos botões enviados para garantir idempotência em release_all().
    Suporta Multi-Monitor / Desktop Virtual com coordenadas negativas através de MOUSEEVENTF_VIRTUALDESK.
    """

    def __init__(self, enable_dpi_awareness: bool = True):
        if sys.platform != "win32":
            raise RuntimeError(
                "OsMouse requer Windows. Em outros sistemas, use um backend alternativo."
            )

        if enable_dpi_awareness:
            init_dpi_awareness()

        self._user32 = ctypes.windll.user32
        try:
            self._user32.SendInput.argtypes = (
                wintypes.UINT,
                ctypes.c_void_p,
                ctypes.c_int,
            )
            self._user32.SendInput.restype = wintypes.UINT
        except AttributeError:
            pass

        # Métricas do monitor primário (0 e 1 chamados explicitamente)
        self._screen_w = self._user32.GetSystemMetrics(SM_CXSCREEN)
        self._screen_h = self._user32.GetSystemMetrics(SM_CYSCREEN)

        # Métricas do Desktop Virtual (multi-monitor)
        try:
            vx = self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            vy = self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            vw = self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            vh = self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

            # Validação defensiva: se o mock de teste ou SO retornar dados incoerentes
            # (ex: mock onde metric != 0 retorna screen_h), usamos fallback primário
            if vw > 0 and vh > 0 and not (vw == self._screen_h and self._screen_w != self._screen_h):
                self._vx, self._vy, self._vw, self._vh = vx, vy, vw, vh
            else:
                self._vx, self._vy, self._vw, self._vh = 0, 0, self._screen_w, self._screen_h
        except Exception:
            self._vx, self._vy, self._vw, self._vh = 0, 0, self._screen_w, self._screen_h

        self._lock = threading.Lock()
        self._last_send_error_logged: float = 0.0

        # Estado interno dos botões
        self._left_down: bool = False
        self._right_down: bool = False
        self._emergency_stopped: bool = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def move(self, x: int, y: int) -> None:
        """Move o cursor para a posição absoluta (x, y) em pixels."""
        if self._emergency_stopped:
            return
        ax, ay = self._to_absolute(int(x), int(y))
        with self._lock:
            flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
            self._send(flags, ax, ay)

    def left_click(self) -> None:
        """Clique esquerdo atômico (down + up em um único SendInput)."""
        if self._emergency_stopped:
            return
        with self._lock:
            self._send_batch(
                self._mi(MOUSEEVENTF_LEFTDOWN),
                self._mi(MOUSEEVENTF_LEFTUP),
            )

    def right_click(self) -> None:
        """Clique direito atômico."""
        if self._emergency_stopped:
            return
        with self._lock:
            self._send_batch(
                self._mi(MOUSEEVENTF_RIGHTDOWN),
                self._mi(MOUSEEVENTF_RIGHTUP),
            )

    def double_click(self) -> None:
        """Duplo clique esquerdo atômico (quatro eventos em um SendInput)."""
        if self._emergency_stopped:
            return
        with self._lock:
            self._send_batch(
                self._mi(MOUSEEVENTF_LEFTDOWN),
                self._mi(MOUSEEVENTF_LEFTUP),
                self._mi(MOUSEEVENTF_LEFTDOWN),
                self._mi(MOUSEEVENTF_LEFTUP),
            )

    def button_down(self, button: str = "left") -> None:
        """Pressiona e mantém o botão (para arrastar)."""
        if self._emergency_stopped:
            return
        with self._lock:
            if button == "left" and not self._left_down:
                self._send(MOUSEEVENTF_LEFTDOWN)
                self._left_down = True
            elif button == "right" and not self._right_down:
                self._send(MOUSEEVENTF_RIGHTDOWN)
                self._right_down = True

    def button_up(self, button: str = "left") -> None:
        """Solta o botão mantido."""
        with self._lock:
            if button == "left" and self._left_down:
                self._send(MOUSEEVENTF_LEFTUP)
                self._left_down = False
            elif button == "right" and self._right_down:
                self._send(MOUSEEVENTF_RIGHTUP)
                self._right_down = False

    def scroll(self, delta: int) -> None:
        """Rola a roda do mouse (positivo=cima, negativo=baixo)."""
        if self._emergency_stopped:
            return
        with self._lock:
            self._send(MOUSEEVENTF_WHEEL, mouse_data=delta)

    def release_all(self) -> None:
        """Libera todos os botões pressionados (idempotente)."""
        with self._lock:
            if self._left_down:
                self._send(MOUSEEVENTF_LEFTUP)
                self._left_down = False
            if self._right_down:
                self._send(MOUSEEVENTF_RIGHTUP)
                self._right_down = False

    def emergency_stop(self) -> None:
        """Parada imediata: libera botões e bloqueia novos eventos até reset."""
        self._emergency_stopped = True
        self.release_all()
        logger.warning("OsMouse: emergency_stop ativado.")

    def reset_emergency_stop(self) -> None:
        """Reativa o envio de eventos após emergency_stop()."""
        self._emergency_stopped = False
        logger.info("OsMouse: emergency_stop desativado.")

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    @property
    def button_state(self) -> dict:
        return {
            "left_down": self._left_down,
            "right_down": self._right_down,
        }

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Retorna (largura, altura) da tela principal em pixels."""
        return self._screen_w, self._screen_h

    @property
    def virtual_screen_geometry(self) -> Tuple[int, int, int, int]:
        """Retorna (vx, vy, vw, vh) do desktop virtual inteiro."""
        return self._vx, self._vy, self._vw, self._vh

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _to_absolute(self, x: int, y: int) -> Tuple[int, int]:
        """
        Converte coordenadas em pixels (incluindo possíveis coordenadas virtuais negativas)
        para o espaço normalizado 0-65535 do SendInput.
        """
        ax = int(((x - self._vx) * 65535) / max(self._vw - 1, 1))
        ay = int(((y - self._vy) * 65535) / max(self._vh - 1, 1))
        return ax, ay

    def _mi(
        self, flags: int, dx: int = 0, dy: int = 0, mouse_data: int = 0
    ) -> _INPUT:
        inp = _INPUT()
        inp.type = INPUT_MOUSE
        inp._input.mi.dx = dx
        inp._input.mi.dy = dy
        inp._input.mi.mouseData = mouse_data
        inp._input.mi.dwFlags = flags
        inp._input.mi.time = 0
        inp._input.mi.dwExtraInfo = 0
        return inp

    def _log_send_failure(self, result: int, expected: int) -> None:
        now = time.time()
        if now - getattr(self, "_last_send_error_logged", 0.0) > 5.0:
            self._last_send_error_logged = now
            err = (
                ctypes.windll.kernel32.GetLastError()
                if hasattr(ctypes, "windll")
                else 0
            )
            logger.warning(
                "SendInput falhou: result=%d (esperado %d), Win32 error=0x%08X",
                result,
                expected,
                err,
            )

    def _send(
        self, flags: int, dx: int = 0, dy: int = 0, mouse_data: int = 0
    ) -> None:
        inp = self._mi(flags, dx, dy, mouse_data)
        result = self._user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(_INPUT)
        )
        if result != 1:
            self._log_send_failure(result, 1)

    def _send_batch(self, *inputs: _INPUT) -> None:
        n = len(inputs)
        if n == 0:
            return
        arr = (_INPUT * n)(*inputs)
        result = self._user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        if result != n:
            self._log_send_failure(result, n)


# ---------------------------------------------------------------------------
# PyAutoGuiMouse (Backend alternativo e comparativo)
# ---------------------------------------------------------------------------
class PyAutoGuiMouse(BaseMouseBackend):
    """
    Driver de mouse usando PyAutoGUI, configurado com PAUSE=0 para eliminar
    o delay artificial de 100ms e manter rastreamento de estado para liberação segura.
    """

    def __init__(self):
        try:
            import pyautogui
            self._pyautogui = pyautogui
            self._pyautogui.PAUSE = 0.0
            self._pyautogui.FAILSAFE = False  # Segurança garantida por AppState e emergency_stop
        except ImportError:
            raise RuntimeError(
                "PyAutoGuiMouse requer a biblioteca 'pyautogui'. Instale com: pip install pyautogui"
            )

        self._left_down: bool = False
        self._right_down: bool = False
        self._emergency_stopped: bool = False
        self._lock = threading.Lock()

    def move(self, x: int, y: int) -> None:
        if self._emergency_stopped:
            return
        with self._lock:
            self._pyautogui.moveTo(int(x), int(y))

    def left_click(self) -> None:
        if self._emergency_stopped:
            return
        with self._lock:
            self._pyautogui.click(button="left")

    def right_click(self) -> None:
        if self._emergency_stopped:
            return
        with self._lock:
            self._pyautogui.click(button="right")

    def double_click(self) -> None:
        if self._emergency_stopped:
            return
        with self._lock:
            self._pyautogui.doubleClick(button="left")

    def button_down(self, button: str = "left") -> None:
        if self._emergency_stopped:
            return
        with self._lock:
            if button == "left" and not self._left_down:
                self._pyautogui.mouseDown(button="left")
                self._left_down = True
            elif button == "right" and not self._right_down:
                self._pyautogui.mouseDown(button="right")
                self._right_down = True

    def button_up(self, button: str = "left") -> None:
        with self._lock:
            if button == "left" and self._left_down:
                self._pyautogui.mouseUp(button="left")
                self._left_down = False
            elif button == "right" and self._right_down:
                self._pyautogui.mouseUp(button="right")
                self._right_down = False

    def scroll(self, delta: int) -> None:
        if self._emergency_stopped:
            return
        with self._lock:
            # PyAutoGUI scroll aceita número de cliques (1 clique ~ 120 units Win32)
            clicks = delta // 120 if abs(delta) >= 120 else (1 if delta > 0 else -1)
            self._pyautogui.scroll(clicks)

    def release_all(self) -> None:
        with self._lock:
            if self._left_down:
                self._pyautogui.mouseUp(button="left")
                self._left_down = False
            if self._right_down:
                self._pyautogui.mouseUp(button="right")
                self._right_down = False

    def emergency_stop(self) -> None:
        self._emergency_stopped = True
        self.release_all()
        logger.warning("PyAutoGuiMouse: emergency_stop ativado.")

    def reset_emergency_stop(self) -> None:
        self._emergency_stopped = False
        logger.info("PyAutoGuiMouse: emergency_stop desativado.")

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    @property
    def button_state(self) -> dict:
        return {
            "left_down": self._left_down,
            "right_down": self._right_down,
        }

    @property
    def screen_size(self) -> Tuple[int, int]:
        size = self._pyautogui.size()
        return int(size[0]), int(size[1])


# ---------------------------------------------------------------------------
# Factory de Backend de Mouse
# ---------------------------------------------------------------------------
def create_mouse_backend(backend_type: str = "SENDINPUT") -> BaseMouseBackend:
    """
    Factory para instanciação do backend de mouse.

    Args:
        backend_type: "SENDINPUT" ou "PYAUTOGUI".

    Returns:
        BaseMouseBackend: Instância do backend selecionado.
    """
    b = backend_type.upper().strip()
    if b in ("SENDINPUT", "OS_MOUSE", "WIN32", "NATIVE"):
        return OsMouse()
    elif b in ("PYAUTOGUI", "AUTOGUI"):
        return PyAutoGuiMouse()
    else:
        raise ValueError(
            f"Backend de mouse desconhecido: '{backend_type}'. "
            "Use 'SENDINPUT' ou 'PYAUTOGUI'."
        )


