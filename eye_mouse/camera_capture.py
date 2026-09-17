"""
camera_capture.py — Abstração de captura de câmera de baixa latência para Windows.

Características:
  - Suporta backends Windows (DirectShow cv2.CAP_DSHOW, Media Foundation cv2.CAP_MSMF, cv2.CAP_ANY).
  - Buffer mínimo (single-slot latest frame): produtor descarta frames antigos se o consumidor
    estiver ocupado, garantindo latência mínima e processamento do frame mais recente.
  - Verifica e registra parâmetros solicitados vs efetivamente aceitos pela câmera.
  - Trata CAP_PROP_BUFFERSIZE defensivamente sem presumir suporte universal.
  - Função de diagnóstico e teste de combinações suportadas (resoluções, FPS, formatos).
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from frame_data import FramePacket

logger = logging.getLogger(__name__)

# Mapeamento de nomes de backend para constantes OpenCV
BACKEND_MAP: Dict[str, int] = {
    "DSHOW": cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 700,
    "MSMF": cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else 1400,
    "ANY": cv2.CAP_ANY if hasattr(cv2, "CAP_ANY") else 0,
}


def fourcc_to_str(fourcc_val: float) -> str:
    """Converte valor float de FOURCC para string legível de 4 caracteres."""
    try:
        val = int(fourcc_val)
        return "".join([chr((val >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return "UNKNOWN"


def str_to_fourcc(codec_str: str) -> int:
    """Converte string de codec (ex: 'MJPG') para FourCC int."""
    if len(codec_str) != 4:
        return 0
    return cv2.VideoWriter_fourcc(*codec_str)


class CameraCapture:
    """
    Captura de câmera multi-threaded com entrega estrita do último frame.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: str = "DSHOW",
        fourcc: Optional[str] = "MJPG",
        buffer_size: int = 1,
        autofocus: Optional[bool] = None,
        exposure: Optional[float] = None,
    ):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.backend_name = backend.upper()
        self.requested_fourcc = fourcc
        self.buffer_size = buffer_size
        self.autofocus = autofocus
        self.exposure = exposure

        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # Buffer de frame mais recente (zero-delay)
        self._latest_packet: Optional[FramePacket] = None
        self._frame_counter: int = 0
        self._dropped_frames: int = 0
        self._captured_frames: int = 0

        # Metadados e diagnóstico real
        self.metadata: Dict[str, Any] = {}
        self.rejected_configurations: List[str] = []

        self._init_camera()

    def _init_camera(self) -> None:
        """Inicializa e configura o objeto VideoCapture com os parâmetros solicitados."""
        backend_code = BACKEND_MAP.get(self.backend_name, cv2.CAP_ANY)
        logger.info(
            "Abrindo câmera %d com backend %s (%d)...",
            self.camera_index, self.backend_name, backend_code
        )

        try:
            self.cap = cv2.VideoCapture(self.camera_index, backend_code)
        except Exception as exc:
            logger.warning("Falha ao abrir com backend %s: %s. Tentando CAP_ANY...", self.backend_name, exc)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
            self.backend_name = "ANY"

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera (index {self.camera_index}, backend {self.backend_name})."
            )

        # 1. Configurar FourCC
        if self.requested_fourcc:
            cc = str_to_fourcc(self.requested_fourcc)
            if cc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cc)

        # 2. Configurar resolução
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        # 3. Configurar FPS solicitado
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        # 4. Configurar buffer interno defensivamente (alguns backends ignoram ou falham)
        try:
            res_buf = self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            if not res_buf:
                self.rejected_configurations.append("CAP_PROP_BUFFERSIZE rejeitado ou não suportado pelo backend")
        except Exception as exc:
            self.rejected_configurations.append(f"CAP_PROP_BUFFERSIZE falhou: {exc}")

        # 5. Configurar autofocus e exposição se especificados
        if self.autofocus is not None:
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
            except Exception:
                self.rejected_configurations.append("CAP_PROP_AUTOFOCUS não suportado")

        if self.exposure is not None:
            try:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            except Exception:
                self.rejected_configurations.append("CAP_PROP_EXPOSURE não suportado")

        # 6. Auditar valores efetivamente aceitos pela câmera
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC))

        if actual_w != self.requested_width or actual_h != self.requested_height:
            self.rejected_configurations.append(
                f"Resolução solicitada {self.requested_width}x{self.requested_height} "
                f"ignorada; câmera fixou em {actual_w}x{actual_h}"
            )

        if self.requested_fps > 0 and abs(actual_fps - self.requested_fps) > 5.0 and actual_fps > 0:
            self.rejected_configurations.append(
                f"FPS solicitado {self.requested_fps} ignorado; câmera reporta {actual_fps:.1f}"
            )

        self.metadata = {
            "camera_index": self.camera_index,
            "backend": self.backend_name,
            "actual_width": actual_w,
            "actual_height": actual_h,
            "actual_fps": actual_fps,
            "actual_fourcc": actual_fourcc,
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "requested_fps": self.requested_fps,
            "rejected_configs": list(self.rejected_configurations),
        }

        logger.info(
            "Câmera inicializada: %dx%d @ %.1f FPS (FourCC: %s, Backend: %s)",
            actual_w, actual_h, actual_fps, actual_fourcc, self.backend_name
        )
        if self.rejected_configurations:
            for rej in self.rejected_configurations:
                logger.warning("Configuração de câmera ajustada/rejeitada: %s", rej)

    def start(self) -> None:
        """Inicia a thread de captura contínua."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self._thread.start()

    def _capture_loop(self) -> None:
        """Loop contínuo de captura: sobrescreve o último frame para latência zero."""
        consecutive_failures = 0
        while self._running:
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            capture_ts = time.perf_counter()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    logger.error("Câmera desconectada ou falhas excessivas de leitura.")
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            self._captured_frames += 1
            self._frame_counter += 1

            packet = FramePacket(
                frame_id=self._frame_counter,
                capture_timestamp=capture_ts,
                image=frame,
                camera_metadata=self.metadata,
            )

            with self._lock:
                if self._latest_packet is not None:
                    # O frame anterior não foi consumido a tempo -> descartado
                    self._dropped_frames += 1
                self._latest_packet = packet

            self._new_frame_event.set()

    def get_latest_frame(self, timeout: Optional[float] = 1.0) -> Optional[FramePacket]:
        """
        Retorna o frame mais recente capturado, consumindo-o.

        Args:
            timeout: Tempo limite em segundos para aguardar novo frame.

        Returns:
            FramePacket ou None se timeout.
        """
        if self._new_frame_event.wait(timeout=timeout):
            with self._lock:
                self._new_frame_event.clear()
                packet = self._latest_packet
                self._latest_packet = None
                return packet
        return None

    def release(self) -> None:
        """Encerra a thread e libera os recursos da câmera."""
        self._running = False
        self._new_frame_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        logger.info("Recursos de captura liberados.")

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def captured_frames(self) -> int:
        return self._captured_frames

    @property
    def is_running(self) -> bool:
        return self._running and bool(self._thread and self._thread.is_alive())

    @property
    def is_connected(self) -> bool:
        return self.is_running and bool(self.cap and self.cap.isOpened())


def probe_camera_configurations(camera_index: int = 0) -> Dict[str, Any]:
    """
    Sonda a câmera testando backends (DSHOW, MSMF), resoluções (320x240, 640x480, 1280x720)
    e mede FPS real entregue por 15 frames em cada modo.
    """
    results: Dict[str, Any] = {"camera_index": camera_index, "tests": []}
    resolutions = [(320, 240), (640, 480), (1280, 720)]
    backends = ["DSHOW", "MSMF"]

    for b_name in backends:
        b_code = BACKEND_MAP.get(b_name, cv2.CAP_ANY)
        for w, h in resolutions:
            test_info: Dict[str, Any] = {
                "backend": b_name,
                "requested_res": f"{w}x{h}",
                "success": False,
            }
            try:
                cap = cv2.VideoCapture(camera_index, b_code)
                if not cap.isOpened():
                    test_info["error"] = "Falha ao abrir VideoCapture"
                    results["tests"].append(test_info)
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, 30)

                act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                test_info["actual_res"] = f"{act_w}x{act_h}"

                # Ler 15 frames para medir FPS real
                t0 = time.perf_counter()
                frames_read = 0
                for _ in range(15):
                    ret, _ = cap.read()
                    if ret:
                        frames_read += 1
                t1 = time.perf_counter()

                cap.release()

                elapsed = t1 - t0
                test_info["fps_real"] = round(frames_read / elapsed, 2) if elapsed > 0 else 0
                test_info["frames_read"] = frames_read
                test_info["success"] = frames_read > 0
            except Exception as exc:
                test_info["error"] = str(exc)

            results["tests"].append(test_info)

    return results
