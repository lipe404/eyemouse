import pyautogui
import threading
try:
    import winsound
except ImportError:
    winsound = None

from utils.smoothing import SmoothingFilter
from config import SCREEN_MARGIN, EMA_ALPHA


class MouseController:
    """
    Controla o cursor do mouse e ações de clique.

    Utiliza a biblioteca pyautogui para mover o cursor e realizar cliques.
    Implementa suavização de movimento para evitar tremores e feedback sonoro
    para ações de clique.
    """

    def __init__(self):
        """Inicializa o controlador do mouse."""
        pyautogui.FAILSAFE = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = SmoothingFilter()
        self.smoothing.set_alpha(EMA_ALPHA)
        self.is_dragging = False

    def _play_sound(self, frequency=1000, duration=100):
        """
        Toca um beep em uma thread separada para não bloquear o mouse.

        Args:
            frequency (int): Frequência do som em Hz.
            duration (int): Duração do som em milissegundos.
        """
        if winsound:
            def run():
                try:
                    winsound.Beep(frequency, duration)
                except Exception:
                    pass
            threading.Thread(target=run, daemon=True).start()

    def move(self, x, y):
        """
        Move o mouse para as coordenadas (x, y) com suavização.

        Aplica um filtro de suavização exponencial e limita o movimento
        às margens da tela.

        Args:
            x (int): Coordenada X alvo.
            y (int): Coordenada Y alvo.
        """
        # Aplicar suavização
        smooth_x, smooth_y = self.smoothing.update(x, y)

        # Limitar às bordas da tela com margem
        final_x = max(SCREEN_MARGIN, min(self.screen_w - SCREEN_MARGIN, smooth_x))
        final_y = max(SCREEN_MARGIN, min(self.screen_h - SCREEN_MARGIN, smooth_y))

        pyautogui.moveTo(final_x, final_y)

    def click(self, button="left"):
        """
        Realiza um clique do mouse.

        Args:
            button (str): Botão a ser clicado ('left' ou 'right').
        """
        pyautogui.click(button=button)
        # Feedback sonoro (Melhoria 10)
        if button == "left":
            self._play_sound(1000, 50)
        else:
            self._play_sound(500, 50) # Som mais grave para direito

    def double_click(self):
        """Realiza um clique duplo do mouse."""
        pyautogui.doubleClick()
        self._play_sound(1500, 50) # Som mais agudo

    def start_drag(self):
        """Inicia a ação de arrastar (segurar botão esquerdo)."""
        if not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True
            self._play_sound(800, 200) # Som longo para indicar 'segurando'

    def stop_drag(self):
        """Finaliza a ação de arrastar (soltar botão esquerdo)."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
            self._play_sound(600, 100) # Som de soltar

    def set_smoothing_alpha(self, alpha):
        """
        Define o fator alpha de suavização.

        Args:
            alpha (float): Fator de suavização (0.0 a 1.0).
        """
        self.smoothing.set_alpha(alpha)
