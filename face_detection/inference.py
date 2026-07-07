"""Face-detection inference on top of an OpenVINO compiled model.

:class:`FaceDetector` is intentionally decoupled from *how* the model was
compiled -- it receives a ready :class:`openvino.CompiledModel` from the
:class:`~device_manager.DeviceManager`. That separation is what makes runtime
device switching cheap: when the device changes, we hand the detector a freshly
compiled model via :meth:`FaceDetector.set_compiled_model` and it keeps running.

Model contract (``face-detection-0200``, an SSD-MobileNetV2 detector)
--------------------------------------------------------------------
* Input : 1 x 3 x H x W, BGR, ``uint8`` pixel values (H = W = 256 by default,
          but we read the real shape from the model to stay robust).
* Output: 1 x 1 x N x 7, where each of the N rows is
          ``[image_id, label, confidence, x_min, y_min, x_max, y_max]`` with the
          box coordinates normalised to ``[0, 1]``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import openvino as ov

from .logger import get_logger
from .utils import Stopwatch

_log = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single detected face with pixel-space coordinates.

    Coordinates are absolute pixels in the *original* frame so downstream
    consumers (logging today, a GUI overlay tomorrow) need no extra scaling.
    """

    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class InferenceResult:
    """Detections plus the time it took to produce them."""

    detections: List[Detection]
    inference_ms: float


class InferenceError(RuntimeError):
    """Raised when a single inference call fails at runtime."""


class FaceDetector:
    """Runs preprocessing, inference and postprocessing for one frame at a time.

    A lock guards the compiled model so the fallback/monitor threads may swap in
    a new model (after a device switch) while the inference thread is mid-flight.
    """

    def __init__(
        self, compiled_model: ov.CompiledModel, confidence_threshold: float
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._lock = threading.Lock()
        self._bind(compiled_model)

    # ------------------------------------------------------------------
    # Model (re)binding -- used on start-up and on every device switch
    # ------------------------------------------------------------------
    def _bind(self, compiled_model: ov.CompiledModel) -> None:
        """Attach a compiled model and cache its input geometry."""

        self._compiled = compiled_model
        self._request = compiled_model.create_infer_request()
        self._input = compiled_model.input(0)
        self._output = compiled_model.output(0)

        # Expected input as N, C, H, W. Reading it from the model keeps us
        # correct even if a different-resolution variant is supplied.
        n, c, h, w = list(self._input.shape)
        self._input_h = int(h)
        self._input_w = int(w)
        _log.debug("Detector bound to input %dx%d (C=%d)", self._input_w, self._input_h, c)

    def set_compiled_model(self, compiled_model: ov.CompiledModel) -> None:
        """Swap the underlying model atomically (called after a device switch)."""

        with self._lock:
            self._bind(compiled_model)
        _log.info("FaceDetector rebound to a new compiled model")

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize to the model input and lay out as NCHW.

        ``face-detection-0200`` consumes raw BGR ``uint8`` values, so no mean /
        scale normalisation is applied -- just resize and transpose.
        """

        resized = cv2.resize(frame, (self._input_w, self._input_h))
        # HWC -> CHW, then add the batch dimension -> NCHW.
        chw = resized.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0)

    def postprocess(
        self, raw_output: np.ndarray, frame_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Turn the raw SSD output into pixel-space :class:`Detection` objects."""

        frame_h, frame_w = frame_shape
        detections: List[Detection] = []

        # Output is [1, 1, N, 7]; collapse the leading batch/anchor dims.
        for row in raw_output.reshape(-1, 7):
            # SSD row layout: [image_id, label, confidence, x_min, y_min, x_max, y_max]
            image_id, confidence = row[0], float(row[2])
            # image_id < 0 marks the end of valid detections in SSD output.
            if image_id < 0 or confidence < self._confidence_threshold:
                continue

            x_min = int(np.clip(row[3], 0.0, 1.0) * frame_w)
            y_min = int(np.clip(row[4], 0.0, 1.0) * frame_h)
            x_max = int(np.clip(row[5], 0.0, 1.0) * frame_w)
            y_max = int(np.clip(row[6], 0.0, 1.0) * frame_h)
            detections.append(
                Detection(confidence, x_min, y_min, x_max, y_max)
            )
        return detections

    def infer(self, frame: np.ndarray) -> InferenceResult:
        """Run the full pipeline for one frame.

        Raises :class:`InferenceError` on any runtime failure so the fallback
        controller can decide whether to switch devices.
        """

        frame_h, frame_w = frame.shape[:2]
        try:
            with self._lock:
                blob = self.preprocess(frame)
                with Stopwatch() as sw:
                    self._request.infer({self._input: blob})
                    raw = self._request.get_output_tensor(0).data
                # Copy out of the tensor before releasing the lock.
                raw = np.array(raw, copy=True)
        except Exception as exc:  # noqa: BLE001 - normalise runtime failures
            raise InferenceError(f"Inference failed: {exc}") from exc

        detections = self.postprocess(raw, (frame_h, frame_w))
        return InferenceResult(detections=detections, inference_ms=sw.elapsed_ms)
