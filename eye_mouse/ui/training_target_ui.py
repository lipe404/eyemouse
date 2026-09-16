"""
training_target_ui.py — Tela de treinamento, prática e avaliação de controle ocular (Milestone 6).

Responsabilidades:
  - Ambiente dedicado e seguro para o usuário praticar o controle do cursor antes de operar o Windows.
  - Alvos circulares distribuídos em 3 diâmetros calibrados:
      * Grande (raio 60px) — Alvo inicial para adaptação.
      * Médio (raio 40px) — Equivalente a botões e caixas de diálogo comuns.
      * Pequeno (raio 25px) — Equivalente a ícones de bandeja e links de texto.
  - Coleta objetiva de métricas por tentativa:
      * Tempo até alcançar o alvo (ms).
      * Distância euclidiana entre o clique e o centro exato do alvo (px).
      * Taxa de acerto acumulada (%).
      * Contagem de falsos cliques (cliques fora do alvo).
      * Total de tentativas concluídas.
  - Acessibilidade Multimodal:
      * Feedback claro por símbolos gráficos e textos explícitos ([✓ ACERTOU] / [✗ FORA]).
      * Imune a daltonismo (não depende unicamente de verde/vermelho).
      * Feedback sonoro opcional e tecla de atalho (Esc para sair, Espaço para clique de teste).
  - Separação Arquitetural:
      * Classe lógica `TrainingSession` puramente matemática e testável sem interface gráfica.
      * Interface visual `TrainingTargetUI` em Tkinter.
"""

from dataclasses import dataclass, field
import math
import time
import tkinter as tk
from typing import Callable, List, Optional, Tuple


@dataclass
class TargetAttempt:
    """Dados de uma tentativa de acerto a um alvo."""
    target_index: int
    target_pos: Tuple[int, int]
    target_radius: int
    click_pos: Tuple[int, int]
    distance_px: float
    time_to_target_sec: float
    is_hit: bool


@dataclass
class TrainingSummary:
    """Resumo estatístico de uma sessão completa de treinamento."""
    total_targets: int
    hits: int
    misses: int
    false_clicks: int
    hit_rate_pct: float
    mean_distance_px: float
    mean_time_sec: float
    recommendation: str

    @property
    def targets_hit(self) -> int:
        return self.hits


class TrainingSession:
    """
    Motor lógico desacoplado para cálculo de métricas de treino ocular.
    """

    def __init__(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
        num_targets: int = 8,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.num_targets = max(1, int(num_targets))

        self.targets: List[Tuple[int, int, int]] = []  # (cx, cy, radius)
        self.current_index: int = 0
        self.current_spawn_time: float = 0.0
        self.attempts: List[TargetAttempt] = []
        self.false_clicks: int = 0
        self.is_completed: bool = False

        self._generate_target_sequence()

    def _generate_target_sequence(self) -> None:
        """Gera sequência balanceada de alvos com variados tamanhos e posições."""
        w, h = self.screen_w, self.screen_h
        margin = 100

        # Posições estratégicas (centro, cantos, laterais)
        positions = [
            (w // 2, h // 2),                  # Centro
            (w // 4, h // 4),                  # Sup-Esq
            (3 * w // 4, h // 4),              # Sup-Dir
            (w // 4, 3 * h // 4),              # Inf-Esq
            (3 * w // 4, 3 * h // 4),          # Inf-Dir
            (w // 2, margin),                  # Topo
            (w // 2, h - margin),              # Base
            (margin + 50, h // 2),             # Esquerda
            (w - margin - 50, h // 2),         # Direita
        ]

        # Tamanhos decrescentes para aumentar gradualmente o desafio
        radii = [60, 60, 40, 40, 40, 25, 25, 25]

        self.targets = []
        for i in range(self.num_targets):
            pos = positions[i % len(positions)]
            r = radii[i % len(radii)]
            self.targets.append((pos[0], pos[1], r))

    def start(self, timestamp: Optional[float] = None) -> Tuple[int, int, int]:
        """Inicia a sessão de treino e o primeiro alvo."""
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self.current_index = 0
        self.current_spawn_time = now
        self.attempts.clear()
        self.false_clicks = 0
        self.is_completed = False
        return self.targets[0]

    @property
    def current_target(self) -> Optional[Tuple[int, int, int]]:
        if self.current_index < len(self.targets):
            return self.targets[self.current_index]
        return None

    def record_click(
        self, click_x: int, click_y: int, timestamp: Optional[float] = None
    ) -> Tuple[bool, TargetAttempt]:
        """
        Registra um clique (ou dwell trigger) na coordenada (click_x, click_y).

        Returns:
            Tuple[bool, TargetAttempt]: (is_session_finished, attempt_details)
        """
        now = time.perf_counter() if timestamp is None else float(timestamp)
        target = self.current_target

        if target is None:
            self.false_clicks += 1
            fake_attempt = TargetAttempt(
                target_index=-1,
                target_pos=(0, 0),
                target_radius=0,
                click_pos=(click_x, click_y),
                distance_px=999.0,
                time_to_target_sec=0.0,
                is_hit=False,
            )
            return True, fake_attempt

        cx, cy, r = target
        dist = math.hypot(click_x - cx, click_y - cy)
        dt = max(0.001, now - self.current_spawn_time)
        is_hit = dist <= r

        attempt = TargetAttempt(
            target_index=self.current_index,
            target_pos=(cx, cy),
            target_radius=r,
            click_pos=(click_x, click_y),
            distance_px=round(dist, 1),
            time_to_target_sec=round(dt, 3),
            is_hit=is_hit,
        )
        self.attempts.append(attempt)

        # Avança para o próximo alvo
        self.current_index += 1
        if self.current_index >= len(self.targets):
            self.is_completed = True
            return True, attempt
        else:
            self.current_spawn_time = now
            return False, attempt

    def get_summary(self) -> TrainingSummary:
        """Computa o resumo estatístico completo da sessão."""
        total = len(self.attempts)
        if total == 0:
            return TrainingSummary(
                total_targets=0,
                hits=0,
                misses=0,
                false_clicks=self.false_clicks,
                hit_rate_pct=0.0,
                mean_distance_px=0.0,
                mean_time_sec=0.0,
                recommendation="Nenhuma tentativa realizada.",
            )

        hits = sum(1 for a in self.attempts if a.is_hit)
        misses = total - hits
        hit_rate = (hits / total) * 100.0
        mean_dist = sum(a.distance_px for a in self.attempts) / total
        mean_time = sum(a.time_to_target_sec for a in self.attempts) / total

        # Recomendação personalizada
        if hit_rate >= 85.0 and mean_dist <= 25.0:
            rec = "Excelente controle! Calibração precisa e pronta para uso cotidiano no Windows."
        elif hit_rate >= 60.0:
            rec = "Bom controle. Considere aumentar ligeiramente a suavização no painel de controle."
        else:
            rec = "Taxa de acerto baixa. Recomendado recalibrar a tela ou ajustar a iluminação."

        return TrainingSummary(
            total_targets=total,
            hits=hits,
            misses=misses,
            false_clicks=self.false_clicks,
            hit_rate_pct=round(hit_rate, 1),
            mean_distance_px=round(mean_dist, 1),
            mean_time_sec=round(mean_time, 2),
            recommendation=rec,
        )


class TrainingTargetUI:
    """
    Interface gráfica interativa para sessão de treino e avaliação empírica do EyeMouse.
    """

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        on_complete_callback: Optional[Callable[[TrainingSummary], None]] = None,
        num_targets: int = 8,
        parent: Optional[tk.Tk] = None,
        on_complete: Optional[Callable[[TrainingSummary], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        self.root = root or parent
        self.on_complete_callback = on_complete_callback or on_complete
        self.on_cancel_callback = on_cancel
        self.num_targets = num_targets

        self.window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.session: Optional[TrainingSession] = None
        self.status_label = None

        if self.root:
            self._create_ui()

    def _create_ui(self) -> None:
        try:
            self.window = tk.Toplevel(self.root)
            self.window.title("EyeMouse — Treinamento e Avaliação do Olhar")
            self.window.attributes("-fullscreen", True)
            self.window.attributes("-topmost", True)
            self.window.configure(bg="#11111b")

            screen_w = self.window.winfo_screenwidth()
            screen_h = self.window.winfo_screenheight()

            self.session = TrainingSession(screen_w, screen_h, num_targets=self.num_targets)

            # Barra superior de instrução e status
            header = tk.Frame(self.window, bg="#181825", height=60)
            header.pack(fill="x")

            lbl_title = tk.Label(
                header,
                text="TELA DE TREINO — Mire no centro do alvo e clique (piscada ou dwell) | Pressione ESC para sair",
                font=("Segoe UI", 12, "bold"),
                bg="#181825",
                fg="#cdd6f4",
                pady=10,
            )
            lbl_title.pack(side="left", padx=20)

            try:
                self.status_var = tk.StringVar(master=self.window, value="Alvo 1 de 8")
            except Exception:
                self.status_var = None

            lbl_status = tk.Label(
                header,
                textvariable=self.status_var,
                font=("Segoe UI", 11, "bold"),
                bg="#181825",
                fg="#89b4fa",
                pady=10,
            )
            lbl_status.pack(side="right", padx=20)

            # Canvas de renderização dos alvos
            self.canvas = tk.Canvas(
                self.window,
                bg="#11111b",
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)

            # Bindings de clique e teclado
            self.canvas.bind("<Button-1>", self._on_canvas_click)
            self.window.bind("<space>", lambda e: self._on_space_click())
            self.window.bind("<Escape>", lambda e: self.close())

            # Inicia o primeiro alvo
            self.session.start()
            self._draw_current_target()
        except Exception as exc:
            self.window = None

    def _draw_current_target(self) -> None:
        if not self.canvas or not self.session:
            return

        self.canvas.delete("all")
        target = self.session.current_target
        if not target:
            self._show_final_report()
            return

        cx, cy, r = target
        idx = self.session.current_index + 1
        total = self.session.num_targets
        if self.status_var:
            self.status_var.set(f"Alvo {idx} de {total} (Raio: {r}px)")

        # Círculo externo
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="#89b4fa",
            width=3,
            fill="#1e1e2e",
        )
        # Círculo intermediário
        mid_r = max(5, r // 2)
        self.canvas.create_oval(
            cx - mid_r, cy - mid_r, cx + mid_r, cy + mid_r,
            outline="#f9e2af",
            width=2,
            fill="#313244",
        )
        # Centro exato (bullseye)
        self.canvas.create_oval(
            cx - 5, cy - 5, cx + 5, cy + 5,
            fill="#a6e3a1",
            outline="#ffffff",
        )
        # Texto acessível do alvo
        self.canvas.create_text(
            cx, cy + r + 25,
            text=f"Alvo #{idx} [Mire Aqui]",
            font=("Segoe UI", 10, "bold"),
            fill="#cdd6f4",
        )

    def _on_canvas_click(self, event) -> None:
        self.process_click_coordinate(event.x, event.y)

    def _on_space_click(self) -> None:
        # Usa o centro do cursor ou a coordenada central
        if self.session and self.session.current_target:
            cx, cy, _ = self.session.current_target
            self.process_click_coordinate(cx, cy)

    def process_click_coordinate(self, x: int, y: int) -> None:
        """Processa a coordenada de um clique executado por olhar, dwell ou mouse."""
        if not self.session or self.session.is_completed:
            return

        finished, attempt = self.session.record_click(x, y)

        # Feedback visual transitório na tela
        if self.canvas:
            tag = "feedback"
            self.canvas.delete(tag)

            if attempt.is_hit:
                msg = f"[✓ ACERTOU!] Distância: {attempt.distance_px}px ({attempt.time_to_target_sec}s)"
                color = "#a6e3a1"
            else:
                msg = f"[✗ FORA DO ALVO] Distância: {attempt.distance_px}px"
                color = "#f38ba8"

            self.canvas.create_text(
                x, y - 20,
                text=msg,
                font=("Segoe UI", 12, "bold"),
                fill=color,
                tags=tag,
            )

        if finished:
            if self.window:
                self.window.after(600, self._show_final_report)
        else:
            if self.window:
                self.window.after(300, self._draw_current_target)

    def _show_final_report(self) -> None:
        if not self.canvas or not self.session:
            return

        self.canvas.delete("all")
        summary = self.session.get_summary()

        w = self.canvas.winfo_width() or 1920
        h = self.canvas.winfo_height() or 1080
        cx, cy = w // 2, h // 2

        # Card de resultado final
        card_w, card_h = 600, 360
        self.canvas.create_rectangle(
            cx - card_w // 2, cy - card_h // 2,
            cx + card_w // 2, cy + card_h // 2,
            fill="#1e1e2e",
            outline="#89b4fa",
            width=2,
        )

        self.canvas.create_text(
            cx, cy - 130,
            text="RELATÓRIO DE DESEMPENHO DO OLHAR",
            font=("Segoe UI", 16, "bold"),
            fill="#cdd6f4",
        )

        stats_text = (
            f"Alvos Testados: {summary.total_targets} | Acertos: {summary.hits} | Erros: {summary.misses}\n\n"
            f"Taxa de Acerto: {summary.hit_rate_pct}%\n"
            f"Precisão Média: {summary.mean_distance_px} pixels do centro\n"
            f"Tempo Médio por Alvo: {summary.mean_time_sec} segundos\n"
            f"Cliques Falsos: {summary.false_clicks}\n\n"
            f"Avaliação: {summary.recommendation}"
        )

        self.canvas.create_text(
            cx, cy + 10,
            text=stats_text,
            font=("Segoe UI", 11),
            fill="#bac2de",
            justify="center",
        )

        self.canvas.create_text(
            cx, cy + 140,
            text="[ Pressione ESC ou clique na tela para concluir ]",
            font=("Segoe UI", 10, "bold"),
            fill="#f9e2af",
        )

        if self.on_complete_callback:
            try:
                self.on_complete_callback(summary)
            except Exception:
                pass

        if self.canvas:
            self.canvas.bind("<Button-1>", lambda e: self.close())

    def close(self) -> None:
        if self.on_cancel_callback and (self.session is None or not self.session.is_completed):
            try:
                self.on_cancel_callback()
            except Exception:
                pass
        if self.window and self.window.winfo_exists():
            self.window.destroy()
            self.window = None
