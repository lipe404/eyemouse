"""
benchmark_gaze.py — Ferramenta de comparação quantitativa dos modelos de olhar:
  Model A: Coordenadas absolutas originais.
  Model B: Posição relativa das íris (normalizada pela geometria ocular).
  Model C: Posição relativa + compensação de pose facial (Yaw, Pitch, Roll).

Avalia o desempenho sobre dados de TREINO e de VALIDAÇÃO (Holdout) sob condições
de movimentos involuntários de cabeça (translação e rotação).
"""
from __future__ import annotations

import argparse
import math
import numpy as np
from typing import Dict, List, Tuple

from calibration_models import (
    PolynomialCalibrationModel,
    RidgeCalibrationModel,
    evaluate_calibration_metrics,
)


def generate_synthetic_dataset(
    num_targets: int = 16,
    samples_per_target: int = 10,
    head_movement: bool = True,
    noise_std: float = 0.005,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> Dict[str, np.ndarray]:
    """
    Gera dataset sintético controlando o olhar em direção à tela e simulando
    movimentos involuntários de translação e rotação da cabeça.
    """
    np.random.seed(42)

    # Grade de alvos na tela
    grid_n = int(math.sqrt(num_targets))
    xs = np.linspace(100, screen_w - 100, grid_n)
    ys = np.linspace(100, screen_h - 100, grid_n)
    targets = [(int(x), int(y)) for y in ys for x in xs]

    data_model_a = []
    data_model_b = []
    data_model_c = []
    screen_coords = []

    for sx, sy in targets:
        # Posição angular nominal do olhar no olho
        norm_x = (sx - screen_w / 2.0) / (screen_w / 2.0) * 0.35 + 0.5
        norm_y = (sy - screen_h / 2.0) / (screen_h / 2.0) * 0.25 + 0.5

        for _ in range(samples_per_target):
            # Movimento involuntário da cabeça
            if head_movement:
                # Translação da cabeça (usuário oscila ~25 pixels na câmera)
                head_tx = np.random.normal(0, 0.04)
                head_ty = np.random.normal(0, 0.03)
                # Rotação da cabeça (yaw e pitch involuntários ~2 a 4 graus)
                yaw_deg = float(np.random.normal(0, 3.5))
                pitch_deg = float(np.random.normal(0, 2.5))
                roll_deg = float(np.random.normal(0, 1.5))
            else:
                head_tx, head_ty = 0.0, 0.0
                yaw_deg, pitch_deg, roll_deg = 0.0, 0.0, 0.0

            noise = np.random.normal(0, noise_std, 2)

            # Model B: Posição relativa da íris (o olho se move relativamente,
            # quase imune à translação da cabeça!)
            rel_iris_x = norm_x + noise[0]
            rel_iris_y = norm_y + noise[1]

            # Model A: Coordenadas absolutas da íris na imagem da câmera
            # Sofre diretamente com a translação da cabeça:
            abs_iris_x = rel_iris_x * 0.15 + 0.45 + head_tx
            abs_iris_y = rel_iris_y * 0.15 + 0.45 + head_ty

            # Model C: Posição relativa + variáveis de pose
            feat_a = [abs_iris_x, abs_iris_y]
            feat_b = [rel_iris_x, rel_iris_y]
            feat_c = [
                rel_iris_x, rel_iris_y,
                yaw_deg / 50.0, pitch_deg / 50.0, roll_deg / 50.0,
                0.5 + head_tx, 0.5 + head_ty,
            ]

            data_model_a.append(feat_a)
            data_model_b.append(feat_b)
            data_model_c.append(feat_c)
            screen_coords.append([sx, sy])

    return {
        "A": np.array(data_model_a, dtype=np.float64),
        "B": np.array(data_model_b, dtype=np.float64),
        "C": np.array(data_model_c, dtype=np.float64),
        "Y": np.array(screen_coords, dtype=np.float64),
    }


def compare_models(head_movement: bool = True) -> None:
    """Executa comparação científica dos três modelos em treino e teste (holdout)."""
    data = generate_synthetic_dataset(num_targets=16, samples_per_target=10, head_movement=head_movement)
    Y = data["Y"]
    N = len(Y)

    # Holdout de 25% separado antes do treino
    indices = np.arange(N)
    np.random.seed(123)
    np.random.shuffle(indices)

    split = int(0.75 * N)
    train_idx, val_idx = indices[:split], indices[split:]

    Y_train, Y_val = Y[train_idx], Y[val_idx]

    print("\n" + "=" * 80)
    print(f" COMPARAÇÃO DE MODELOS DE OLHAR (Movimento de Cabeça: {'ATIVO' if head_movement else 'NENHUM'})")
    print("=" * 80)
    print(f"Total de Amostras: {N} | Treino: {len(train_idx)} | Validação (Holdout): {len(val_idx)}")
    print("-" * 80)
    print(f"{'Modelo':<10} | {'RMSE Treino':<12} | {'RMSE Val (px)':<14} | {'Mediana Val':<12} | {'P95 Val (px)':<12} | {'% Diag':<8}")
    print("-" * 80)

    for name in ["A", "B", "C"]:
        X = data[name]
        X_train, X_val = X[train_idx], X[val_idx]

        model = RidgeCalibrationModel(alpha_l2=1e-3, use_polynomial=(name in ["A", "B"]))
        model.fit(X_train, Y_train)

        m_train = evaluate_calibration_metrics(model, X_train, Y_train)
        m_val = evaluate_calibration_metrics(model, X_val, Y_val)

        desc = {
            "A": "Model A (Absoluto)",
            "B": "Model B (Relativo)",
            "C": "Model C (Relativo + Pose)",
        }[name]

        print(
            f"{desc:<25} | {m_train.rmse_px:9.1f} px | {m_val.rmse_px:11.1f} px | "
            f"{m_val.median_error_px:9.1f} px | {m_val.p95_error_px:9.1f} px | {m_val.relative_diag_pct:6.2f}%"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparação de Modelos de Olhar")
    parser.add_argument("--no-movement", action="store_true", help="Sem movimento de cabeça")
    args = parser.parse_args()

    compare_models(head_movement=not args.no_movement)
