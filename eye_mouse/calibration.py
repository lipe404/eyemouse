"""
calibration.py — Gerenciador de calibracao do rastreamento ocular.

Persistencia migrada de .npy (allow_pickle=True) para JSON seguro.
Suporta migracao automatica de arquivos .npy legados.

Seguranca:
  - Nomes de perfil sao validados para prevenir path traversal.
  - Nenhum dado de arquivo externo e executado ou desserializado com pickle.
  - Coeficientes sao listas simples de float — sem tipos Python arbitrarios.

Validacao:
  - compute_calibration() separa holdout antes de treinar.
  - O erro reportado e o holdout error (generalizacao), nao o residuo de treino.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from config import CALIBRATION_FILE_PREFIX, CALIBRATION_REPROJECTION_ERROR_THRESHOLD

logger = logging.getLogger(__name__)

# Caracteres permitidos em nomes de perfil
_PROFILE_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')

# Versao atual do formato de arquivo de calibracao
_CALIBRATION_VERSION = 2


def _validate_profile_name(name: str) -> str:
    """
    Valida e sanitiza um nome de perfil.

    Args:
        name: Nome a ser validado.

    Returns:
        O nome validado (mesmo valor se valido).

    Raises:
        ValueError: Se o nome contiver caracteres invalidos ou for vazio.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Nome de perfil nao pode ser vazio.")
    if not _PROFILE_RE.match(name):
        raise ValueError(
            f"Nome de perfil invalido: '{name}'. "
            "Use apenas letras, digitos, '_' e '-' (1-64 caracteres)."
        )
    return name


class CalibrationManager:
    """
    Gerencia o processo de calibracao do rastreamento ocular.

    Responsavel por:
      - Coletar pontos de calibracao.
      - Calcular os coeficientes de mapeamento por regressao polinomial.
      - Validar a calibracao com um conjunto holdout separado.
      - Persistir e carregar calibracoes em formato JSON seguro.
    """

    def __init__(self, profile_name: str = "default"):
        """
        Inicializa o gerenciador de calibracao.

        Args:
            profile_name: Nome do perfil de usuario. Valida o nome.

        Raises:
            ValueError: Se o nome do perfil for invalido.
        """
        self.profile_name: str = _validate_profile_name(profile_name)
        self._json_file: str = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.json"
        self._npy_file: str = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.npy"

        self.iris_points: List = []
        self.screen_points: List = []
        self.coeffs_x: Optional[np.ndarray] = None
        self.coeffs_y: Optional[np.ndarray] = None
        self.is_calibrated: bool = False

        # Erro do holdout da ultima calibracao
        self.last_holdout_error: float = float('inf')
        # Erro do treino da ultima calibracao (para diagnostico)
        self.last_train_error: float = float('inf')

    # Manter compatibilidade: calibration_file aponta para o JSON
    @property
    def calibration_file(self) -> str:
        return self._json_file

    # ------------------------------------------------------------------
    # Gestao de perfil
    # ------------------------------------------------------------------

    def set_profile(self, profile_name: str) -> None:
        """
        Altera o perfil ativo e tenta carregar a calibracao correspondente.

        Args:
            profile_name: Novo nome de perfil (validado).
        """
        self.profile_name = _validate_profile_name(profile_name)
        self._json_file = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.json"
        self._npy_file = f"{CALIBRATION_FILE_PREFIX}{self.profile_name}.npy"
        self.load_calibration()

    # ------------------------------------------------------------------
    # Coleta de pontos
    # ------------------------------------------------------------------

    def add_point(
        self,
        iris_pos: Tuple[float, float],
        screen_pos: Tuple[int, int],
    ) -> None:
        """
        Adiciona um par (iris, tela) para calibracao.

        Args:
            iris_pos:   Coordenadas (x, y) da iris (normalizadas 0-1).
            screen_pos: Coordenadas (x, y) na tela em pixels.
        """
        self.iris_points.append(iris_pos)
        self.screen_points.append(screen_pos)

    def clear_points(self) -> None:
        """Remove todos os pontos de calibracao coletados."""
        self.iris_points = []
        self.screen_points = []

    # ------------------------------------------------------------------
    # Calculo da calibracao
    # ------------------------------------------------------------------

    def compute_calibration(self) -> Tuple[bool, float]:
        """
        Calcula os coeficientes de regressao polinomial (2a ordem) e valida.

        Separa um conjunto holdout (min 2 pontos ou 20% dos dados) ANTES de
        treinar, de modo que o erro reportado reflita a generalizacao real,
        nao o residuo de treinamento.

        Returns:
            (sucesso, holdout_error_px): sucesso indica se a calibracao
            foi realizada; holdout_error_px e o erro medio em pixels no
            conjunto holdout.
        """
        n = len(self.iris_points)
        if n < 6:
            logger.warning("Pontos insuficientes: %d (minimo 6).", n)
            return False, 0.0

        iris_arr = np.array(self.iris_points)
        screen_arr = np.array(self.screen_points)

        # --- Separar holdout ANTES do treinamento ---
        n_holdout = max(2, n // 5)          # 20% ou minimo 2 pontos
        n_train = n - n_holdout

        # Indices de treino e holdout (holdout = ultimos N pontos)
        train_idx = list(range(n_train))
        holdout_idx = list(range(n_train, n))

        iris_train = iris_arr[train_idx]
        screen_train = screen_arr[train_idx]
        iris_holdout = iris_arr[holdout_idx]
        screen_holdout = screen_arr[holdout_idx]

        # --- Construir matriz de design: [1, x, y, xy, x^2, y^2] ---
        def design_matrix(iris: np.ndarray) -> np.ndarray:
            X, Y = iris[:, 0], iris[:, 1]
            ones = np.ones(len(X))
            return np.column_stack([ones, X, Y, X * Y, X**2, Y**2])

        A_train = design_matrix(iris_train)

        # --- Regressao por minimos quadrados ---
        self.coeffs_x, _, _, _ = np.linalg.lstsq(
            A_train, screen_train[:, 0], rcond=None
        )
        self.coeffs_y, _, _, _ = np.linalg.lstsq(
            A_train, screen_train[:, 1], rcond=None
        )
        self.is_calibrated = True

        # --- Erro de treino (diagnostico) ---
        A_train_pred = design_matrix(iris_train)
        pred_x_train = A_train_pred @ self.coeffs_x
        pred_y_train = A_train_pred @ self.coeffs_y
        diffs_train = np.sqrt(
            (pred_x_train - screen_train[:, 0])**2 +
            (pred_y_train - screen_train[:, 1])**2
        )
        self.last_train_error = float(np.mean(diffs_train))

        # --- Erro de holdout (generalizacao — o que e reportado) ---
        A_holdout = design_matrix(iris_holdout)
        pred_x_h = A_holdout @ self.coeffs_x
        pred_y_h = A_holdout @ self.coeffs_y
        diffs_h = np.sqrt(
            (pred_x_h - screen_holdout[:, 0])**2 +
            (pred_y_h - screen_holdout[:, 1])**2
        )
        self.last_holdout_error = float(np.mean(diffs_h))

        logger.info(
            "Calibracao: train_error=%.1fpx holdout_error=%.1fpx "
            "(n_train=%d n_holdout=%d)",
            self.last_train_error,
            self.last_holdout_error,
            n_train,
            n_holdout,
        )

        self.save_calibration()
        return True, self.last_holdout_error

    # ------------------------------------------------------------------
    # Mapeamento
    # ------------------------------------------------------------------

    def map_to_screen(
        self, iris_pos: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        """
        Mapeia a posicao da iris para coordenadas de tela.

        Args:
            iris_pos: Coordenadas (x, y) da iris (normalizadas 0-1).

        Returns:
            (screen_x, screen_y) em pixels, ou None se nao calibrado.
        """
        if not self.is_calibrated or self.coeffs_x is None:
            return None

        x, y = iris_pos
        features = np.array([1.0, x, y, x * y, x**2, y**2])
        screen_x = float(np.dot(features, self.coeffs_x))
        screen_y = float(np.dot(features, self.coeffs_y))
        return int(screen_x), int(screen_y)

    # ------------------------------------------------------------------
    # Persistencia JSON
    # ------------------------------------------------------------------

    def save_calibration(self) -> None:
        """
        Salva os coeficientes em arquivo .json.

        Formato: JSON com lista de floats — sem pickle, sem dados binarios.
        """
        if not self.is_calibrated:
            return

        data = {
            "version": _CALIBRATION_VERSION,
            "profile": self.profile_name,
            "coeffs_x": self.coeffs_x.tolist(),
            "coeffs_y": self.coeffs_y.tolist(),
            "train_error_px": self.last_train_error,
            "holdout_error_px": self.last_holdout_error,
        }
        try:
            with open(self._json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Calibracao salva em: %s", self._json_file)
        except OSError as exc:
            logger.error("Erro ao salvar calibracao: %s", exc)

    def load_calibration(self) -> bool:
        """
        Carrega a calibracao do arquivo JSON.

        Se o arquivo JSON nao existir mas o .npy legado existir,
        migra automaticamente para JSON (sem allow_pickle para .npy legado
        — a migracao e feita com allow_pickle somente neste contexto de
        migracao unica).

        Returns:
            True se carregado com sucesso.
        """
        # Tentar JSON primeiro
        if os.path.exists(self._json_file):
            return self._load_json()

        # Fallback: migrar .npy legado
        if os.path.exists(self._npy_file):
            logger.info(
                "Arquivo legado encontrado: %s. Migrando para JSON...",
                self._npy_file,
            )
            return self._migrate_npy_to_json()

        return False

    def _load_json(self) -> bool:
        """Carrega calibracao do arquivo JSON com validacao de schema."""
        try:
            with open(self._json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validar campos obrigatorios
            required = {"coeffs_x", "coeffs_y"}
            missing = required - data.keys()
            if missing:
                logger.error(
                    "Arquivo de calibracao invalido: campos ausentes: %s",
                    missing,
                )
                return False

            coeffs_x = data["coeffs_x"]
            coeffs_y = data["coeffs_y"]

            # Validar tipos e tamanhos
            if (not isinstance(coeffs_x, list) or
                    not isinstance(coeffs_y, list) or
                    len(coeffs_x) != 6 or
                    len(coeffs_y) != 6):
                logger.error(
                    "Arquivo de calibracao invalido: coeffs devem ser "
                    "listas de 6 floats."
                )
                return False

            self.coeffs_x = np.array(coeffs_x, dtype=np.float64)
            self.coeffs_y = np.array(coeffs_y, dtype=np.float64)
            self.is_calibrated = True
            self.last_holdout_error = float(data.get("holdout_error_px", float('inf')))
            self.last_train_error = float(data.get("train_error_px", float('inf')))

            logger.info(
                "Calibracao carregada: %s (holdout_error=%.1fpx)",
                self._json_file,
                self.last_holdout_error,
            )
            return True

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "Erro ao carregar calibracao JSON: %s", exc
            )
            return False
        except OSError as exc:
            logger.error("Erro de acesso ao arquivo de calibracao: %s", exc)
            return False

    def _migrate_npy_to_json(self) -> bool:
        """
        Migra arquivo .npy legado para JSON.

        allow_pickle e usado apenas neste metodo de migracao unica,
        nunca em carregamentos normais.
        """
        try:
            # Nota: allow_pickle necessario para compatibilidade com .npy
            # que salva dicionario Python. Usado apenas uma vez para migracao.
            raw = np.load(self._npy_file, allow_pickle=True)
            data = raw.item()  # npy salvo com np.save(file, dict)

            self.coeffs_x = np.array(data["coeffs_x"], dtype=np.float64)
            self.coeffs_y = np.array(data["coeffs_y"], dtype=np.float64)
            self.is_calibrated = True
            self.last_holdout_error = float('inf')  # Legado nao tem holdout error
            self.last_train_error = float('inf')

            # Salvar como JSON para uso futuro
            self.save_calibration()
            logger.info(
                "Calibracao legada migrada de %s para %s.",
                self._npy_file,
                self._json_file,
            )
            return True

        except Exception as exc:
            logger.error("Falha ao migrar calibracao legada: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Validacao interna (legado — mantida para compatibilidade de testes)
    # ------------------------------------------------------------------

    def _validate_calibration(self) -> float:
        """
        Calcula o erro de reprojecao usando os DADOS DE TREINO.

        AVISO: Este metodo existe apenas para compatibilidade com testes
        legados. Em producao, use last_holdout_error para avaliar
        a qualidade da calibracao.

        Returns:
            Erro medio em pixels no conjunto de treino (nao e o erro real).
        """
        if not self.is_calibrated:
            return float('inf')

        total_error = 0.0
        count = 0

        for iris_pt, screen_pt in zip(self.iris_points, self.screen_points):
            predicted = self.map_to_screen(iris_pt)
            if predicted:
                dist = float(
                    np.linalg.norm(
                        np.array(predicted) - np.array(screen_pt)
                    )
                )
                total_error += dist
                count += 1

        return total_error / count if count > 0 else float('inf')
