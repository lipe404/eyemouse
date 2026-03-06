import pyautogui
from utils.smoothing import SmoothingFilter
from config import SCREEN_MARGIN


class MouseController:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = SmoothingFilter()
        self.is_dragging = False

    def move(self, x, y):
        """Move o mouse para as coordenadas (x, y) com suavização."""
        # Aplicar suavização
        smooth_x, smooth_y = self.smoothing.update(x, y)

        # Limitar às bordas da tela com margem
        final_x = max(SCREEN_MARGIN, min(self.screen_w - SCREEN_MARGIN, smooth_x))
        final_y = max(SCREEN_MARGIN, min(self.screen_h - SCREEN_MARGIN, smooth_y))

        pyautogui.moveTo(final_x, final_y)

    def click(self, button="left"):
        pyautogui.click(button=button)

    def double_click(self):
        pyautogui.doubleClick()

    def start_drag(self):
        if not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True

    def stop_drag(self):
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False

    def set_smoothing_alpha(self, alpha):
        self.smoothing.set_alpha(alpha)
