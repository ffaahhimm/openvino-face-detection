"""The inference pipeline: the consumer half of the capture/inference split.

:class:`InferenceWorker` is the second of the two required threads (the first is
:class:`~camera.CameraThread`). It pulls the latest frame from the
:class:`~camera.FrameStore`, runs the :class:`~inference.FaceDetector`, updates
:class:`PipelineStats`, and publishes results through an event so any consumer
(console reporter today, GUI overlay tomorrow) can react.

On an inference failure it delegates to the :class:`~fallback.FallbackController`
rather than crashing -- resilience is a first-class behaviour, not an
afterthought.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from .camera import CapturedFrame, FrameStore
from .events import Event
from .fallback import FallbackController
from .inference import FaceDetector, InferenceError
from .logger import get_logger
from .utils import FPSCounter, MovingAverage

_log = get_logger(__name__)

# Callable returning the current device string (kept as a callback so the worker
# stays decoupled from DeviceManager internals).
DeviceProvider = Callable[[], str]


@dataclass(frozen=True)
class StatsSnapshot:
    """Immutable copy of pipeline stats, safe to hand to any thread/consumer."""

    fps: float
    avg_inference_ms: float
    last_inference_ms: float
    frames_processed: int
    faces_detected: int
    errors: int
    device: str


class PipelineStats:
    """Thread-safe accumulator of pipeline performance counters."""

    def __init__(self, device_provider: DeviceProvider, window: int = 30) -> None:
        self._device_provider = device_provider
        self._fps = FPSCounter(window=window)
        self._inference_ms = MovingAverage(window=window)
        self._lock = threading.Lock()
        self._frames_processed = 0
        self._faces_detected = 0
        self._errors = 0

    def record_success(self, inference_ms: float, face_count: int) -> None:
        with self._lock:
            self._fps.tick()
            self._inference_ms.add(inference_ms)
            self._frames_processed += 1
            self._faces_detected = face_count

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            return StatsSnapshot(
                fps=self._fps.fps,
                avg_inference_ms=self._inference_ms.value,
                last_inference_ms=self._inference_ms.last or 0.0,
                frames_processed=self._frames_processed,
                faces_detected=self._faces_detected,
                errors=self._errors,
                device=self._device_provider(),
            )


class InferenceWorker(threading.Thread):
    """Consumes frames, runs detection, publishes results, drives fallback."""

    def __init__(
        self,
        store: FrameStore,
        detector: FaceDetector,
        stats: PipelineStats,
        fallback: FallbackController,
        idle_sleep_sec: float = 0.001,
    ) -> None:
        super().__init__(name="InferenceWorker", daemon=True)
        self._store = store
        self._detector = detector
        self._stats = stats
        self._fallback = fallback
        self._idle_sleep_sec = idle_sleep_sec
        self._stop_event = threading.Event()

        # Emitted after every processed frame: (frame, detections, stats).
        self.on_result = Event("inference_result")

    def run(self) -> None:
        _log.info("Inference worker started")
        last_frame_id = 0
        while not self._stop_event.is_set():
            captured = self._store.get()
            # No frame yet, or we already processed this one: wait briefly.
            if captured is None or captured.frame_id == last_frame_id:
                self._stop_event.wait(self._idle_sleep_sec)
                continue
            last_frame_id = captured.frame_id
            self._process(captured)
        _log.info("Inference worker stopped")

    def _process(self, captured: CapturedFrame) -> None:
        try:
            result = self._detector.infer(captured.image)
        except InferenceError as exc:
            self._stats.record_error()
            # Hand the failure to the fallback controller, which may switch
            # devices once the error threshold is exceeded.
            self._fallback.handle_inference_error(exc)
            return

        self._fallback.notify_success()
        self._stats.record_success(result.inference_ms, len(result.detections))
        self.on_result.emit(captured.image, result.detections, self._stats.snapshot())

    def stop(self) -> None:
        self._stop_event.set()
