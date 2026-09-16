"""
action_bar.py — Barra de Ações flutuante e ancorável do EyeMouse (Milestone 5).

Responsabilidades:
  - Disponibiliza visualmente as ações essenciais do mouse:
      * Clique Esquerdo (padrão)
      * Clique Direito (armado para próximo clique)
      * Duplo Clique (armado para próximo clique)
      * Iniciar / Parar Arrasto (Drag & Drop com feedback visual)
      * Modo Rolagem (Scroll Controller toggle)
      * Modo Precisão (Precision Mode toggle)
      * Pausa / Retomada
      * Reposicionamento / Ancoragem (Dock: Top, Bottom, Left, Right)
      * Recolher / Expandir barra
  - Modelo Next-Action:
      * Ao selecionar uma ação (ex: Clique Direito ou Duplo Clique), a barra a mantém
        armada em destaque visual.
      * O próximo clique ou dwell executado na tela executa a ação armada e reverte
        automaticamente para Clique Esquerdo.
  - Acionamento por Dwell ou Clique Direto:
      * Suporta cliques do mouse e detecção de dwell por coordenadas de tela (hit_test).
      * Quando um clique por dwell ocorre sobre a barra, a ação é ativada e não vaza para a janela abaixo.
"""

from enum import Enum
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Tuple

from gesture_engine import MouseAction

logger = logging.getLogger(__name__)


class DockPosition(Enum):
    """Posições de ancoragem da barra de ações na tela."""
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ActionBar:
    """
    Barra de ações visual do EyeMouse com despacho Next-Action e suporte a Dwell.
    """

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        on_action_selected: Optional[Callable[[MouseAction], None]] = None,
        on_pause_toggle: Optional[Callable[[bool], None]] = None,
        on_scroll_toggle: Optional[Callable[[bool], None]] = None,
        on_precision_toggle: Optional[Callable[[bool], None]] = None,
        on_drag_toggle: Optional[Callable[[bool], None]] = None,
    ):
        self.root = root
        self.on_action_selected = on_action_selected
        self.on_pause_toggle = on_pause_toggle
        self.on_scroll_toggle = on_scroll_toggle
        self.on_precision_toggle = on_precision_toggle
        self.on_drag_toggle = on_drag_toggle

        # Estados de ação
        self.armed_action: MouseAction = MouseAction.LEFT_CLICK
        self.is_dragging: bool = False
        self.is_scrolling: bool = False
        self.is_precision: bool = False
        self.is_paused: bool = False
        self.dock_position: DockPosition = DockPosition.TOP
        self.is_collapsed: bool = False

        # Dimensões da tela e da barra
        self.screen_w: int = 1920
        self.screen_h: int = 1080
        self.window_w: int = 760
        self.window_h: int = 65

        # Criação da janela Tkinter (se root fornecido)
        self.window: Optional[tk.Toplevel] = None
        self._buttons: Dict[str, tk.Button] = {}
        self._btn_bounds: Dict[str, Tuple[int, int, int, int]] = {}  # Nome -> (x1, y1, x2, y2) relativos à tela

        if self.root:
            self._create_ui()

    # ------------------------------------------------------------------
    # Construção da Interface
    # ------------------------------------------------------------------

    def _create_ui(self) -> None:
        try:
            self.window = tk.Toplevel(self.root)
            self.window.title("EyeMouse Action Bar")
            self.window.attributes("-topmost", True)
            self.window.resizable(False, False)

            # Estilo ferramenta flutuante
            try:
                self.window.attributes("-toolwindow", True)
            except Exception:
                pass

            self.screen_w = self.window.winfo_screenwidth()
            self.screen_h = self.window.winfo_screenheight()

            # Cores de alto contraste
            self.bg_color = "#1e1e2e"
            self.fg_color = "#cdd6f4"
            self.btn_bg = "#313244"
            self.btn_active = "#f9e2af"
            self.btn_active_fg = "#11111b"
            self.btn_toggle_on = "#89b4fa"

            self.window.configure(bg=self.bg_color)

            # Frame para botões
            self.frame = tk.Frame(self.window, bg=self.bg_color, padx=5, pady=5)
            self.frame.pack(fill="both", expand=True)

            self._build_buttons()
            self._apply_dock_geometry()

            # Protocolo de fechamento: apenas oculta/recolhe
            self.window.protocol("WM_DELETE_WINDOW", self.hide)
        except Exception as exc:
            logger.warning("Falha ao criar interface Tkinter da ActionBar (%s). Modo lógico ativo.", exc)
            self.window = None

    def _build_buttons(self) -> None:
        if not self.frame:
            return

        for child in self.frame.winfo_children():
            child.destroy()
        self._buttons.clear()

        # Definição dos botões da barra
        btn_defs = [
            ("LEFT_CLICK", "Esq (Padrão)", lambda: self.arm_action(MouseAction.LEFT_CLICK)),
            ("RIGHT_CLICK", "Dir", lambda: self.arm_action(MouseAction.RIGHT_CLICK)),
            ("DOUBLE_CLICK", "2x Clique", lambda: self.arm_action(MouseAction.DOUBLE_CLICK)),
            ("DRAG", "Arrastar", self._on_drag_click),
            ("SCROLL", "Rolagem", self._on_scroll_click),
            ("PRECISION", "Precisão", self._on_precision_click),
            ("PAUSE", "Pausar", self._on_pause_click),
            ("DOCK", "Mover", self.cycle_dock),
        ]

        is_vertical = self.dock_position in (DockPosition.LEFT, DockPosition.RIGHT)

        for btn_id, text, cmd in btn_defs:
            btn = tk.Button(
                self.frame,
                text=text,
                command=cmd,
                font=("Segoe UI", 9, "bold"),
                bg=self.btn_bg,
                fg=self.fg_color,
                activebackground=self.btn_active,
                activeforeground=self.btn_active_fg,
                relief="raised",
                bd=2,
                padx=8,
                pady=4,
            )
            if is_vertical:
                btn.pack(fill="x", expand=True, pady=2)
            else:
                btn.pack(side="left", fill="y", expand=True, padx=2)

            self._buttons[btn_id] = btn

        self._refresh_button_styles()

    def _apply_dock_geometry(self) -> None:
        if not self.window:
            return

        pos = self.dock_position
        if pos == DockPosition.TOP:
            w, h = 760, 58
            x = (self.screen_w - w) // 2
            y = 10
        elif pos == DockPosition.BOTTOM:
            w, h = 760, 58
            x = (self.screen_w - w) // 2
            y = max(10, self.screen_h - h - 50)
        elif pos == DockPosition.LEFT:
            w, h = 110, 420
            x = 10
            y = (self.screen_h - h) // 2
        elif pos == DockPosition.RIGHT:
            w, h = 110, 420
            x = max(10, self.screen_w - w - 20)
            y = (self.screen_h - h) // 2

        self.window_w, self.window_h = w, h
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        self.window.update_idletasks()
        self._update_bounds_cache()

    def _update_bounds_cache(self) -> None:
        """Atualiza as coordenadas absolutas de tela de cada botão para hit-testing rápido de Dwell."""
        if not self.window or not self.window.winfo_exists():
            return

        try:
            win_x = self.window.winfo_rootx()
            win_y = self.window.winfo_rooty()

            for btn_id, btn in self._buttons.items():
                bx = btn.winfo_rootx()
                by = btn.winfo_rooty()
                bw = btn.winfo_width()
                bh = btn.winfo_height()
                self._btn_bounds[btn_id] = (bx, by, bx + bw, by + bh)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Disparos e Ações de Botões
    # ------------------------------------------------------------------

    def arm_action(self, action: MouseAction) -> None:
        """Define a próxima ação do mouse a ser executada no próximo clique/dwell."""
        self.armed_action = action
        logger.info("ActionBar: ação armada para %s", action.name)
        self._refresh_button_styles()
        if self.on_action_selected:
            self.on_action_selected(action)

    def consume_armed_action(self) -> MouseAction:
        """
        Retorna a ação armada atual.
        Se for uma ação de tiro único (RIGHT_CLICK ou DOUBLE_CLICK), reverte
        automaticamente para LEFT_CLICK após o consumo.
        """
        action = self.armed_action
        if action in (MouseAction.RIGHT_CLICK, MouseAction.DOUBLE_CLICK):
            self.armed_action = MouseAction.LEFT_CLICK
            self._refresh_button_styles()
            logger.info("ActionBar: ação %s consumida; revertendo para LEFT_CLICK.", action.name)
        return action

    def _on_drag_click(self) -> None:
        new_state = not self.is_dragging
        self.set_dragging(new_state)
        if self.on_drag_toggle:
            self.on_drag_toggle(new_state)

    def _on_scroll_click(self) -> None:
        new_state = not self.is_scrolling
        self.set_scrolling(new_state)
        if self.on_scroll_toggle:
            self.on_scroll_toggle(new_state)

    def _on_precision_click(self) -> None:
        new_state = not self.is_precision
        self.set_precision(new_state)
        if self.on_precision_toggle:
            self.on_precision_toggle(new_state)

    def _on_pause_click(self) -> None:
        new_state = not self.is_paused
        self.set_paused(new_state)
        if self.on_pause_toggle:
            self.on_pause_toggle(new_state)

    def set_dragging(self, is_dragging: bool) -> None:
        self.is_dragging = bool(is_dragging)
        self._refresh_button_styles()

    def set_scrolling(self, is_scrolling: bool) -> None:
        self.is_scrolling = bool(is_scrolling)
        self._refresh_button_styles()

    def set_precision(self, is_precision: bool) -> None:
        self.is_precision = bool(is_precision)
        self._refresh_button_styles()

    def set_paused(self, is_paused: bool) -> None:
        self.is_paused = bool(is_paused)
        self._refresh_button_styles()

    def cycle_dock(self) -> DockPosition:
        """Alterna a posição da barra entre TOP -> BOTTOM -> LEFT -> RIGHT -> TOP."""
        cycle = [
            DockPosition.TOP,
            DockPosition.BOTTOM,
            DockPosition.LEFT,
            DockPosition.RIGHT,
        ]
        curr_idx = cycle.index(self.dock_position)
        self.dock_position = cycle[(curr_idx + 1) % len(cycle)]
        logger.info("ActionBar: ancorada em %s", self.dock_position.name)

        if self.window and self.frame:
            self._build_buttons()
            self._apply_dock_geometry()

        return self.dock_position

    # ------------------------------------------------------------------
    # Atualização Visual dos Estilos
    # ------------------------------------------------------------------

    def _refresh_button_styles(self) -> None:
        if not self._buttons:
            return

        # 1. Clique Esquerdo
        btn_left = self._buttons.get("LEFT_CLICK")
        if btn_left:
            if self.armed_action == MouseAction.LEFT_CLICK:
                btn_left.configure(bg="#22c55e", fg="#ffffff", text="Esq [ATIVO]")
            else:
                btn_left.configure(bg=self.btn_bg, fg=self.fg_color, text="Esq")

        # 2. Clique Direito
        btn_right = self._buttons.get("RIGHT_CLICK")
        if btn_right:
            if self.armed_action == MouseAction.RIGHT_CLICK:
                btn_right.configure(bg="#f59e0b", fg="#000000", text="Dir [ARMADO]")
            else:
                btn_right.configure(bg=self.btn_bg, fg=self.fg_color, text="Dir")

        # 3. Duplo Clique
        btn_dbl = self._buttons.get("DOUBLE_CLICK")
        if btn_dbl:
            if self.armed_action == MouseAction.DOUBLE_CLICK:
                btn_dbl.configure(bg="#f59e0b", fg="#000000", text="2x [ARMADO]")
            else:
                btn_dbl.configure(bg=self.btn_bg, fg=self.fg_color, text="2x Clique")

        # 4. Arrasto (Drag & Drop)
        btn_drag = self._buttons.get("DRAG")
        if btn_drag:
            if self.is_dragging:
                btn_drag.configure(bg="#ef4444", fg="#ffffff", text="Soltar")
            else:
                btn_drag.configure(bg=self.btn_bg, fg=self.fg_color, text="Arrastar")

        # 5. Rolagem
        btn_scroll = self._buttons.get("SCROLL")
        if btn_scroll:
            if self.is_scrolling:
                btn_scroll.configure(bg=self.btn_toggle_on, fg="#000000", text="Rolagem ON")
            else:
                btn_scroll.configure(bg=self.btn_bg, fg=self.fg_color, text="Rolagem")

        # 6. Precisão
        btn_prec = self._buttons.get("PRECISION")
        if btn_prec:
            if self.is_precision:
                btn_prec.configure(bg=self.btn_toggle_on, fg="#000000", text="Precisão ON")
            else:
                btn_prec.configure(bg=self.btn_bg, fg=self.fg_color, text="Precisão")

        # 7. Pausa
        btn_pause = self._buttons.get("PAUSE")
        if btn_pause:
            if self.is_paused:
                btn_pause.configure(bg="#eab308", fg="#000000", text="Retomar")
            else:
                btn_pause.configure(bg=self.btn_bg, fg=self.fg_color, text="Pausar")

    # ------------------------------------------------------------------
    # Dwell Hit-Testing (Detecção de Fixação sobre a Barra)
    # ------------------------------------------------------------------

    def hit_test(self, screen_x: int, screen_y: int) -> Optional[str]:
        """
        Verifica se a coordenada de tela (screen_x, screen_y) está sobre algum botão da barra.
        Retorna o identificador do botão atingido ou None.
        """
        if not self.is_visible():
            return None

        self._update_bounds_cache()

        for btn_id, (x1, y1, x2, y2) in self._btn_bounds.items():
            if x1 <= screen_x <= x2 and y1 <= screen_y <= y2:
                return btn_id

        # Verifica se está no retângulo geral da janela
        if self.window and self.window.winfo_exists():
            wx = self.window.winfo_rootx()
            wy = self.window.winfo_rooty()
            ww = self.window.winfo_width()
            wh = self.window.winfo_height()
            if wx <= screen_x <= (wx + ww) and wy <= screen_y <= (wy + wh):
                return "WINDOW"

        return None

    def handle_dwell_click(self, screen_x: int, screen_y: int) -> bool:
        """
        Processa um disparo de dwell click nas coordenadas de tela fornecidas.

        Se o cursor estiver sobre a barra de ações:
          - Aciona o botão correspondente;
          - Retorna True (evita clique pass-through para o Windows).
        Caso contrário, retorna False.
        """
        btn_hit = self.hit_test(screen_x, screen_y)
        if not btn_hit:
            return False

        logger.info("ActionBar: Dwell click detectado no elemento '%s' (%d, %d)", btn_hit, screen_x, screen_y)

        if btn_hit == "LEFT_CLICK":
            self.arm_action(MouseAction.LEFT_CLICK)
        elif btn_hit == "RIGHT_CLICK":
            self.arm_action(MouseAction.RIGHT_CLICK)
        elif btn_hit == "DOUBLE_CLICK":
            self.arm_action(MouseAction.DOUBLE_CLICK)
        elif btn_hit == "DRAG":
            self._on_drag_click()
        elif btn_hit == "SCROLL":
            self._on_scroll_click()
        elif btn_hit == "PRECISION":
            self._on_precision_click()
        elif btn_hit == "PAUSE":
            self._on_pause_click()
        elif btn_hit == "DOCK":
            self.cycle_dock()

        return True

    # ------------------------------------------------------------------
    # Visibilidade e Ciclo de Vida
    # ------------------------------------------------------------------

    def is_visible(self) -> bool:
        if not self.window or not self.window.winfo_exists():
            return False
        return self.window.winfo_viewable() == 1

    def show(self) -> None:
        if self.window and self.window.winfo_exists():
            self.window.deiconify()
            self._apply_dock_geometry()

    def hide(self) -> None:
        if self.window and self.window.winfo_exists():
            self.window.withdraw()

    def destroy(self) -> None:
        if self.window and self.window.winfo_exists():
            self.window.destroy()
            self.window = None
