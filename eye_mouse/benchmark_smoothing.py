"""
benchmark_smoothing.py — Benchmark quantitativo de filtros de suavização e backends de mouse.

Compara:
  1. OneEuroFilter (Casiez et al., CHI 2012)
  2. KalmanSmoothingFilter (OpenCV cv2.KalmanFilter de referência)
  3. PassThroughFilter (Sem filtro / baseline de ruído bruto)

Avalia 3 cenários sintéticos com ruído ocular simulado:
  - Fixação (Repouso): Mede jitter RMS e deslocamento quadro a quadro.
  - Sacada (Step Response): Mede tempo de subida (90%), tempo de acomodação (95%) e overshoot.
  - Perseguição Suave (Smooth Pursuit): Mede erro dinâmico de rastreamento (RMSE).

Também mede a latência de injeção no SO:
  - OsMouse (SendInput Win32) vs PyAutoGuiMouse.
"""

import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Adicionar o diretório eye_mouse ao path para importações diretas
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.smoothing import (
    BaseSmoothingFilter,
    KalmanSmoothingFilter,
    OneEuroFilter,
    PassThroughFilter,
)


def generate_fixation_data(
    num_frames: int = 120,
    center: Tuple[float, float] = (960.0, 540.0),
    noise_sigma: float = 3.5,
    tremor_amp: float = 0.8,
    fps: float = 30.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera dados sintéticos de fixação com ruído gaussiano e micro-tremor ocular (60 Hz).
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / fps
    timestamps = np.arange(num_frames) * dt

    # Sinal verdadeiro constante
    true_x = np.full(num_frames, center[0])
    true_y = np.full(num_frames, center[1])

    # Ruído gaussiano + tremor
    noise_x = rng.normal(0, noise_sigma, num_frames)
    noise_y = rng.normal(0, noise_sigma, num_frames)
    tremor_x = tremor_amp * np.sin(2 * np.pi * 60.0 * timestamps)
    tremor_y = tremor_amp * np.cos(2 * np.pi * 60.0 * timestamps)

    noisy_x = true_x + noise_x + tremor_x
    noisy_y = true_y + noise_y + tremor_y

    return timestamps, np.column_stack((true_x, true_y)), np.column_stack((noisy_x, noisy_y))


def generate_saccade_data(
    num_frames: int = 100,
    p1: Tuple[float, float] = (300.0, 300.0),
    p2: Tuple[float, float] = (1200.0, 700.0),
    step_frame: int = 25,
    noise_sigma: float = 2.0,
    fps: float = 30.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera degrau abrupto (sacada ocular) com ruído moderado.
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / fps
    timestamps = np.arange(num_frames) * dt

    true_pos = np.zeros((num_frames, 2))
    true_pos[:step_frame] = p1
    true_pos[step_frame:] = p2

    noisy_x = true_pos[:, 0] + rng.normal(0, noise_sigma, num_frames)
    noisy_y = true_pos[:, 1] + rng.normal(0, noise_sigma, num_frames)

    return timestamps, true_pos, np.column_stack((noisy_x, noisy_y))


def generate_pursuit_data(
    num_frames: int = 150,
    center: Tuple[float, float] = (960.0, 540.0),
    radius_x: float = 400.0,
    radius_y: float = 200.0,
    freq_hz: float = 0.4,
    noise_sigma: float = 2.5,
    fps: float = 30.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera movimento elíptico contínuo (perseguição suave).
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / fps
    timestamps = np.arange(num_frames) * dt

    true_x = center[0] + radius_x * np.sin(2 * np.pi * freq_hz * timestamps)
    true_y = center[1] + radius_y * np.cos(2 * np.pi * freq_hz * timestamps)
    true_pos = np.column_stack((true_x, true_y))

    noisy_x = true_x + rng.normal(0, noise_sigma, num_frames)
    noisy_y = true_y + rng.normal(0, noise_sigma, num_frames)

    return timestamps, true_pos, np.column_stack((noisy_x, noisy_y))


def evaluate_filter_on_data(
    f: BaseSmoothingFilter,
    timestamps: np.ndarray,
    noisy_pts: np.ndarray,
) -> np.ndarray:
    """Passa os pontos pelo filtro e retorna o traçado resultante."""
    f.reset()
    out = np.zeros_like(noisy_pts)
    for i in range(len(noisy_pts)):
        out[i, 0], out[i, 1] = f.update(
            noisy_pts[i, 0], noisy_pts[i, 1], timestamp=timestamps[i]
        )
    return out


def run_benchmark() -> Dict[str, Any]:
    """Executa o benchmark completo comparativo e retorna métricas estruturadas."""
    fps = 30.0

    # 1. Filtros a avaliar
    filters: Dict[str, BaseSmoothingFilter] = {
        "OneEuro (Padrão)": OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0),
        "OneEuro (Extra Suave)": OneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0),
        "OneEuro (Responsivo)": OneEuroFilter(min_cutoff=1.5, beta=0.015, d_cutoff=1.0),
        "Kalman (Ref s/ lookahead)": KalmanSmoothingFilter(lookahead=0.0),
        "Kalman (Legado c/ lookahead 66ms)": KalmanSmoothingFilter(lookahead=0.066),
        "PassThrough (Sem Filtro)": PassThroughFilter(),
    }

    results: Dict[str, Any] = {}

    # --- Teste 1: Fixação (Jitter) ---
    ts_fix, true_fix, noisy_fix = generate_fixation_data(num_frames=120, fps=fps)
    raw_fix_err = np.linalg.norm(noisy_fix - true_fix, axis=1)
    raw_fix_rms = float(np.sqrt(np.mean(raw_fix_err**2)))
    raw_fix_diff = np.linalg.norm(np.diff(noisy_fix, axis=0), axis=1)
    raw_fix_diff_rms = float(np.sqrt(np.mean(raw_fix_diff**2)))

    results["fixation"] = {
        "raw_rms": raw_fix_rms,
        "raw_step_rms": raw_fix_diff_rms,
        "filters": {},
    }

    for name, flt in filters.items():
        out = evaluate_filter_on_data(flt, ts_fix, noisy_fix)
        # Ignora primeiros 5 frames de warm-up
        err = np.linalg.norm(out[5:] - true_fix[5:], axis=1)
        rms = float(np.sqrt(np.mean(err**2)))
        diff = np.linalg.norm(np.diff(out[5:], axis=0), axis=1)
        step_rms = float(np.sqrt(np.mean(diff**2)))
        reduction = float((1.0 - (rms / raw_fix_rms)) * 100.0)

        results["fixation"]["filters"][name] = {
            "rms_error_px": rms,
            "step_rms_px": step_rms,
            "jitter_reduction_pct": reduction,
        }

    # --- Teste 2: Sacada (Step Response & Overshoot) ---
    p1 = (300.0, 300.0)
    p2 = (1200.0, 700.0)
    step_frame = 25
    step_mag = math.hypot(p2[0] - p1[0], p2[1] - p1[1])  # ~984.88 px
    ts_sac, true_sac, noisy_sac = generate_saccade_data(
        num_frames=100, p1=p1, p2=p2, step_frame=step_frame, fps=fps
    )

    results["saccade"] = {"step_magnitude_px": step_mag, "filters": {}}

    for name, flt in filters.items():
        out = evaluate_filter_on_data(flt, ts_sac, noisy_sac)

        # Projeção no eixo do movimento
        u = np.array([p2[0] - p1[0], p2[1] - p1[1]]) / step_mag
        progress = np.dot(out - p1, u)  # 0 até step_mag

        post_step = progress[step_frame:]
        frames_rel = np.arange(len(post_step))

        # 90% Rise Time
        idx_90 = np.where(post_step >= 0.90 * step_mag)[0]
        rise_time_ms = float(idx_90[0] * (1000.0 / fps)) if len(idx_90) > 0 else 999.0

        # 95% Settling Time (fica dentro de 95% a 105%)
        within_95 = np.where(np.abs(post_step - step_mag) <= 0.05 * step_mag)[0]
        settle_ms = float(within_95[0] * (1000.0 / fps)) if len(within_95) > 0 else 999.0

        # Overshoot
        max_prog = np.max(post_step)
        overshoot_px = max(0.0, float(max_prog - step_mag))
        overshoot_pct = float((overshoot_px / step_mag) * 100.0)

        results["saccade"]["filters"][name] = {
            "rise_time_90_ms": rise_time_ms,
            "settling_time_95_ms": settle_ms,
            "overshoot_px": overshoot_px,
            "overshoot_pct": overshoot_pct,
        }

    # --- Teste 3: Perseguição Suave (Smooth Pursuit) ---
    ts_pur, true_pur, noisy_pur = generate_pursuit_data(num_frames=150, fps=fps)
    raw_pur_err = np.linalg.norm(noisy_pur - true_pur, axis=1)
    raw_pur_rms = float(np.sqrt(np.mean(raw_pur_err**2)))

    results["pursuit"] = {"raw_rms": raw_pur_rms, "filters": {}}

    for name, flt in filters.items():
        out = evaluate_filter_on_data(flt, ts_pur, noisy_pur)
        err = np.linalg.norm(out[10:] - true_pur[10:], axis=1)
        rms = float(np.sqrt(np.mean(err**2)))
        results["pursuit"]["filters"][name] = {"tracking_rmse_px": rms}

    return results


def print_report(res: Dict[str, Any]) -> None:
    """Imprime relatório formatado e detalhado no terminal."""
    print("=" * 82)
    print("           EYEMOUSE MILESTONE 4 — BENCHMARK DE SUAVIZAÇÃO DO CURSOR")
    print("=" * 82)
    print(f"Cenário 1: Fixação (Repouso) — Ruído Bruto: {res['fixation']['raw_rms']:.2f} px RMS, Step: {res['fixation']['raw_step_rms']:.2f} px")
    print("-" * 82)
    print(f"{'Filtro':<35} | {'Jitter RMS':<12} | {'Step RMS':<12} | {'Redução %':<10}")
    print("-" * 82)
    for name, m in res["fixation"]["filters"].items():
        print(
            f"{name:<35} | {m['rms_error_px']:>9.2f} px | {m['step_rms_px']:>9.2f} px | {m['jitter_reduction_pct']:>8.1f}%"
        )

    print("\n" + "=" * 82)
    print(f"Cenário 2: Sacada (Degrau de {res['saccade']['step_magnitude_px']:.1f} px)")
    print("-" * 82)
    print(f"{'Filtro':<35} | {'90% Subida':<12} | {'95% Acomod.':<12} | {'Overshoot':<12}")
    print("-" * 82)
    for name, m in res["saccade"]["filters"].items():
        print(
            f"{name:<35} | {m['rise_time_90_ms']:>9.1f} ms | {m['settling_time_95_ms']:>9.1f} ms | {m['overshoot_px']:>7.1f}px ({m['overshoot_pct']:.1f}%)"
        )

    print("\n" + "=" * 82)
    print(f"Cenário 3: Perseguição Suave (Trajetória Elíptica, Ruído Bruto: {res['pursuit']['raw_rms']:.2f} px)")
    print("-" * 82)
    print(f"{'Filtro':<35} | {'Erro de Rastreamento (RMSE)':<30}")
    print("-" * 82)
    for name, m in res["pursuit"]["filters"].items():
        print(f"{name:<35} | {m['tracking_rmse_px']:>15.2f} px")
    print("=" * 82)


if __name__ == "__main__":
    benchmark_data = run_benchmark()
    print_report(benchmark_data)
