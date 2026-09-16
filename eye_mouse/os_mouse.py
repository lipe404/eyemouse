"""
os_mouse.py — Driver nativo de mouse para Windows via SendInput.

Substitui o pyautogui para eventos de mouse, eliminando o PAUSE artificial
de 100 ms por chamada. Usa a API Win32 SendInput diretamente via ctypes.

Nao altera configuracoes globais do Windows (velocidade do mouse, aceleracao).
Nao depende de permissoes de administrador para eventos basicos de mouse.

Protecao equivalente ao pyautogui.FAILSAFE:
  - AppState machine impede acoes em estados invalidos (main.py).
  - release_all() e chamado em qualquer transicao de estado anormal.
  - emergency_stop() libera botoes imediatamente.
"""
import ctypes
from ctypes import wintypes
import logging
import sys
import threading
import time

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


# ---------------------------------------------------------------------------
# Estruturas Win32 necessarias para SendInput
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
# OsMouse
# ---------------------------------------------------------------------------
class OsMouse:
    """
    Driver de mouse de baixo nivel usando a API Win32 SendInput.

    Thread-safe. Rastreia apenas o estado dos botoes que foram enviados
    para garantir idempotencia em release_all().

    Nao move o cursor real quando emergency_stopped == True.
    """

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError(
                "OsMouse requer Windows. Em outros sistemas, use um backend alternativo."
            )
        self._user32 = ctypes.windll.user32
        # Configurar argtypes e restype de SendInput
        try:
            self._user32.SendInput.argtypes = (
                wintypes.UINT,
                ctypes.c_void_p,
                ctypes.c_int,
            )
            self._user32.SendInput.restype = wintypes.UINT
        except AttributeError:
            pass

        self._screen_w = self._user32.GetSystemMetrics(SM_CXSCREEN)
        self._screen_h = self._user32.GetSystemMetrics(SM_CYSCREEN)
        self._lock = threading.Lock()
        self._last_send_error_logged: float = 0.0

        # Estado interno dos botoes (espelho do que foi enviado ao SO)
        self._left_down: bool = False
        self._right_down: bool = False
        self._emergency_stopped: bool = False

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def move(self, x: int, y: int) -> None:
        """Move o cursor para a posicao absoluta (x, y) em pixels."""
        if self._emergency_stopped:
            return
        ax, ay = self._to_absolute(int(x), int(y))
        with self._lock:
            self._send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)

    def left_click(self) -> None:
        """Clique esquerdo atomico (down + up em um unico SendInput)."""
        if self._emergency_stopped:
            return
        with self._lock:
            self._send_batch(
                self._mi(MOUSEEVENTF_LEFTDOWN),
                self._mi(MOUSEEVENTF_LEFTUP),
            )

    def right_click(self) -> None:
        """Clique direito atomico."""
        if self._emergency_stopped:
            return
        with self._lock:
            self._send_batch(
                self._mi(MOUSEEVENTF_RIGHTDOWN),
                self._mi(MOUSEEVENTF_RIGHTUP),
            )

    def double_click(self) -> None:
        """Duplo clique esquerdo atomico (quatro eventos em um SendInput)."""
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
        """
        Pressiona e mantem o botao (para arrastar).

        Args:
            button: "left" ou "right".
        """
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
        """
        Solta o botao mantido.

        Args:
            button: "left" ou "right".
        """
        with self._lock:
            if button == "left" and self._left_down:
                self._send(MOUSEEVENTF_LEFTUP)
                self._left_down = False
            elif button == "right" and self._right_down:
                self._send(MOUSEEVENTF_RIGHTUP)
                self._right_down = False

    def scroll(self, delta: int) -> None:
        """
        Rola a roda do mouse.

        Args:
            delta: Positivo = cima, negativo = baixo.
                   120 = um dente de scroll padrao do Windows.
        """
        if self._emergency_stopped:
            return
        with self._lock:
            self._send(MOUSEEVENTF_WHEEL, mouse_data=delta)

    def release_all(self) -> None:
        """
        Libera todos os botoes pressionados.

        Idempotente: seguro para multiplas chamadas consecutivas.
        Nao afetado por emergency_stopped (sempre executa).
        """
        with self._lock:
            if self._left_down:
                self._send(MOUSEEVENTF_LEFTUP)
                self._left_down = False
            if self._right_down:
                self._send(MOUSEEVENTF_RIGHTUP)
                self._right_down = False

    def emergency_stop(self) -> None:
        """
        Para imediata: libera botoes e bloqueia novos eventos ate reset.

        Use quando o sistema entrar em estado de erro critico.
        Para retomar, chame reset_emergency_stop().
        """
        self._emergency_stopped = True
        self.release_all()
        logger.warning("OsMouse: emergency_stop ativado.")

    def reset_emergency_stop(self) -> None:
        """Reativa o envio de eventos apos emergency_stop()."""
        self._emergency_stopped = False
        logger.info("OsMouse: emergency_stop desativado.")

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    @property
    def button_state(self) -> dict:
        """Retorna o estado atual dos botoes (para debug/testes)."""
        return {
            "left_down": self._left_down,
            "right_down": self._right_down,
        }

    @property
    def screen_size(self):
        """Retorna (largura, altura) da tela principal em pixels."""
        return self._screen_w, self._screen_h

    # ------------------------------------------------------------------
    # Metodos privados
    # ------------------------------------------------------------------

    def _to_absolute(self, x: int, y: int):
        """Converte coordenadas em pixels para o espaco 0-65535 do SendInput."""
        ax = (x * 65535) // max(self._screen_w - 1, 1)
        ay = (y * 65535) // max(self._screen_h - 1, 1)
        return ax, ay

    def _mi(self, flags: int, dx: int = 0, dy: int = 0,
            mouse_data: int = 0) -> _INPUT:
        """Cria um struct INPUT preenchido para mouse."""
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
        now = time.time() if "time" in globals() else 0.0
        if now - getattr(self, "_last_send_error_logged", 0.0) > 5.0:
            self._last_send_error_logged = now
            err = ctypes.windll.kernel32.GetLastError() if hasattr(ctypes, "windll") else 0
            logger.warning(
                "SendInput falhou: result=%d (esperado %d), Win32 error=0x%08X",
                result, expected, err,
            )

    def _send(self, flags: int, dx: int = 0, dy: int = 0,
              mouse_data: int = 0) -> None:
        """Envia um unico evento via SendInput (ja dentro de _lock)."""
        inp = self._mi(flags, dx, dy, mouse_data)
        result = self._user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(_INPUT)
        )
        if result != 1:
            self._log_send_failure(result, 1)

    def _send_batch(self, *inputs: _INPUT) -> None:
        """Envia multiplos eventos atomicamente em um unico SendInput."""
        n = len(inputs)
        if n == 0:
            return
        arr = (_INPUT * n)(*inputs)
        result = self._user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        if result != n:
            self._log_send_failure(result, n)

