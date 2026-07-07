"""Webcam capture.

Two responsibilities, cleanly separated:

* :class:`Camera`       -- a thin, testable wrapper around ``cv2.VideoCapture``.
* :class:`CameraThread` -- a producer thread that continuously grabs frames and
  publishes the *latest* one into a :class:`FrameStore`.

The pipeline is intentionally "latest frame wins": we never queue up stale
frames, so inference always works on the freshest image and latency stays low
even when the model runs slower than the camera delivers frames.

Failure handling is fully automatic and backend-only -- no dialogs, no buttons:
a failing or disconnected camera is detected, released, and recreated from
scratch (DirectShow preferred, then MSMF) on the capture thread itself, while
the GUI and the inference pipeline keep running untouched and simply resume
when frames flow again.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig
from .logger import get_logger

_log = get_logger(__name__)

# Also lower OpenCV's runtime log level (covers the case where cv2 was imported
# before OPENCV_LOG_LEVEL was set). Best-effort: the API varies across versions.
try:  # pragma: no cover - depends on the installed OpenCV build
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # noqa: BLE001
    pass


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read from."""


@dataclass(frozen=True)
class CapturedFrame:
    """An immutable snapshot handed from the capture thread to consumers."""

    image: np.ndarray
    frame_id: int
    timestamp: float  # time.monotonic() when captured


class FrameStore:
    """Thread-safe single-slot buffer holding only the most recent frame.

    The capture thread calls :meth:`put`; the inference worker calls :meth:`get`
    and compares ``frame_id`` against the last one it processed to detect whether
    a genuinely new frame is available.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[CapturedFrame] = None
        self._counter = 0

    def put(self, image: np.ndarray) -> None:
        with self._lock:
            self._counter += 1
            self._frame = CapturedFrame(
                image=image, frame_id=self._counter, timestamp=time.monotonic()
            )

    def get(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._frame


# Ordered (name, API id) candidates for opening the device. On Windows the
# default MSMF backend is both slow to enumerate and the usual source of
# driver-level read failures, so DirectShow is always tried first; MSMF and
# the auto-selected backend remain as fallbacks for devices DirectShow cannot
# drive. Elsewhere OpenCV's default choice (V4L2, AVFoundation, ...) is right.
if sys.platform == "win32":
    _BACKENDS: List[Tuple[str, int]] = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("default", cv2.CAP_ANY),
    ]
else:
    _BACKENDS = [("default", cv2.CAP_ANY)]

# A freshly opened device must actually deliver a frame within this budget to
# count as usable -- ``isOpened()`` alone often returns True for a dead device.
_PROBE_ATTEMPTS = 3
_PROBE_DEADLINE_SEC = 2.0


class Camera:
    """Minimal wrapper around ``cv2.VideoCapture`` with explicit lifecycle.

    Thread-safety: the handle swap is guarded by a lock, but the blocking
    driver call inside :meth:`read` deliberately runs *outside* it, so
    :meth:`release` can always be called from another thread -- that is how the
    watchdog unhooks a read stuck inside a hung driver.

    Supports use as a context manager so callers cannot forget to release the
    underlying device.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._backend = ""
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._capture is not None

    @property
    def backend(self) -> str:
        """Name of the backend that opened the device ('' while closed)."""

        with self._lock:
            return self._backend

    def open(self) -> None:
        """Open the device, apply the configured settings, and verify it streams.

        Tries each backend in preference order (DirectShow before MSMF on
        Windows) and only accepts one that actually delivers a frame. Used both
        for the initial open and to recreate a fresh ``VideoCapture`` during
        recovery; the resolution/FPS settings are re-applied on every call.
        """

        capture, backend = self._create_capture()
        with self._lock:
            stale, self._capture = self._capture, capture
            self._backend = backend
        if stale is not None:  # defensive: open() over an un-released handle
            stale.release()

        self._warmup()
        _log.info(
            "Camera %d opened via %s backend (%dx%d @ %d fps requested, %dx%d actual)",
            self._config.index,
            backend,
            self._config.width,
            self._config.height,
            self._config.fps,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _create_capture(self) -> Tuple["cv2.VideoCapture", str]:
        index = self._config.index
        for name, api in _BACKENDS:
            _log.info("Trying %s backend...", name)
            try:
                capture = cv2.VideoCapture(index, api)
            except Exception as exc:  # noqa: BLE001 - a broken backend must not stop the ladder
                _log.warning("%s backend raised while opening camera %d: %s", name, index, exc)
                continue
            if not capture.isOpened():
                capture.release()
                continue
            self._apply_settings(capture)
            if not self._probe(capture):
                _log.warning(
                    "%s backend opened camera %d but delivered no frames", name, index
                )
                capture.release()
                continue
            return capture, name
        raise CameraError(
            f"Unable to open camera at index {index} "
            f"(tried: {', '.join(name for name, _ in _BACKENDS)})"
        )

    def _apply_settings(self, capture: "cv2.VideoCapture") -> None:
        """Apply (or restore, after recovery) the configured capture settings."""

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)

    @staticmethod
    def _probe(capture: "cv2.VideoCapture") -> bool:
        """Require a real frame before trusting a device that claims to be open."""

        deadline = time.monotonic() + _PROBE_DEADLINE_SEC
        for _ in range(_PROBE_ATTEMPTS):
            try:
                ok, frame = capture.read()
            except Exception:  # noqa: BLE001 - a throwing device is a dead device
                return False
            if ok and frame is not None:
                return True
            if time.monotonic() >= deadline:
                break
        return False

    def _warmup(self) -> None:
        """Discard the first few frames so auto-exposure/white-balance settle."""

        with self._lock:
            capture = self._capture
        if capture is None:
            return
        for _ in range(max(0, self._config.warmup_frames)):
            try:
                capture.read()
            except Exception:  # noqa: BLE001 - warmup is best-effort
                return

    def read(self) -> np.ndarray:
        """Grab a single frame or raise :class:`CameraError`.

        Any driver-level exception (``cv2.error`` from MSMF is the classic
        case) is normalised into :class:`CameraError`, so callers have a single
        failure type to handle and the capture loop can never be killed by an
        exotic backend error.
        """

        with self._lock:
            capture = self._capture
        if capture is None:
            raise CameraError("Camera is not open")
        try:
            ok, frame = capture.read()
        except Exception as exc:  # noqa: BLE001 - normalise driver faults
            raise CameraError(f"Camera read raised: {exc}") from exc
        if not ok or frame is None:
            raise CameraError("Failed to read frame from camera")
        return frame

    def release(self) -> None:
        """Release the device. Safe to call from any thread, any number of times."""

        with self._lock:
            capture, self._capture = self._capture, None
            self._backend = ""
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - a dead driver may throw; the handle is gone either way
                pass

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class CameraThread(threading.Thread):
    """Background producer that keeps :class:`FrameStore` populated.

    Recovery is fully automatic and confined to this thread:

    * transient read failures are retried quietly with a short backoff;
    * reads failing for longer than ``disconnect_grace_sec`` -- or a device
      that is no longer open at all -- mean the camera is gone: the thread
      releases the dead handle and recreates a fresh ``VideoCapture``
      (DirectShow first) with exponential backoff until frames flow again;
    * a read *hung inside the driver* is spotted by a small watchdog thread,
      which force-releases the device to unblock the stuck call, after which
      the normal recovery path takes over.

    The GUI and the inference worker are never involved: during an outage the
    worker simply sees no new frame ids, and both resume automatically once
    streaming does.
    """

    def __init__(self, camera: Camera, store: FrameStore, config: CameraConfig) -> None:
        super().__init__(name="CameraThread", daemon=True)
        self._camera = camera
        self._store = store
        self._config = config
        self._stop_event = threading.Event()
        # Set while a read is in flight; the watchdog uses it to detect a hang.
        self._read_started_at: Optional[float] = None
        self._read_lock = threading.Lock()

    def run(self) -> None:
        _log.info("Capture thread started")
        if self._config.read_hang_timeout_sec > 0:
            threading.Thread(
                target=self._watchdog_loop, name="CameraWatchdog", daemon=True
            ).start()

        failing_since: Optional[float] = None
        failures = 0
        while not self._stop_event.is_set():
            try:
                frame = self._read_watched()
            except Exception as exc:  # noqa: BLE001 - the capture loop must never die
                failures += 1
                if failing_since is None:
                    failing_since = time.monotonic()
                    _log.warning("Camera read failed: %s", exc)
                grace_expired = (
                    time.monotonic() - failing_since >= self._config.disconnect_grace_sec
                )
                if grace_expired or not self._camera.is_open:
                    self._recover()
                    failing_since, failures = None, 0
                else:
                    self._stop_event.wait(self._config.error_retry_delay_sec)
                continue

            if failing_since is not None:
                _log.info("Camera recovered after %d failed read(s)", failures)
                failing_since, failures = None, 0
            self._store.put(frame)
        _log.info("Capture thread stopped")

    def _read_watched(self) -> np.ndarray:
        """Read one frame while exposing the in-flight window to the watchdog."""

        with self._read_lock:
            self._read_started_at = time.monotonic()
        try:
            return self._camera.read()
        finally:
            with self._read_lock:
                self._read_started_at = None

    def _recover(self) -> None:
        """Blocking recovery loop: rebuild the capture until frames flow again.

        Runs entirely on this thread -- the GUI keeps rendering and the
        inference worker keeps polling throughout. Backs off exponentially
        between attempts so a camera that stays absent for minutes (or hours)
        never spins the CPU, yet a replug is picked up within seconds. All
        waits go through the stop event, so shutdown stays responsive even
        mid-recovery.
        """

        _log.warning("Camera disconnected.")
        self._camera.release()  # drop the dead handle before recreating
        delay = max(0.1, self._config.reconnect_delay_sec)
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            _log.info("Attempting recovery... (attempt %d)", attempt)
            try:
                self._camera.open()
            except Exception as exc:  # noqa: BLE001 - keep trying, whatever the driver throws
                _log.error("Recovery attempt %d failed: %s", attempt, exc)
                if self._stop_event.wait(delay):
                    break
                delay = min(delay * 2.0, self._config.reconnect_max_delay_sec)
                continue
            _log.info("Camera recovered successfully.")
            _log.info("Streaming resumed.")
            return
        _log.info("Recovery abandoned: shutdown requested")

    def _watchdog_loop(self) -> None:
        """Detect a read() stuck inside the driver and unhook it.

        MSMF in particular can block a read indefinitely when the device
        resets mid-call -- from inside the capture thread that is
        indistinguishable from a slow frame, so a second thread has to watch.
        Releasing the capture from here makes the stuck call return with an
        error in the capture thread, whose normal recovery path then rebuilds
        the device. OpenCV does not guarantee concurrent release()+read(), but
        once a read has been wedged past the timeout the only alternative is a
        permanently frozen pipeline, so this is the designed last resort.
        """

        timeout = self._config.read_hang_timeout_sec
        poll = min(1.0, timeout / 2.0)
        while not self._stop_event.wait(poll):
            with self._read_lock:
                started = self._read_started_at
            if started is None or time.monotonic() - started < timeout:
                continue
            _log.error(
                "Camera read hung for %.1fs; force-releasing the device to unblock it",
                time.monotonic() - started,
            )
            with self._read_lock:
                self._read_started_at = None  # one release per hang
            self._camera.release()

    def stop(self) -> None:
        """Signal the loop to exit (call :meth:`join` afterwards)."""

        self._stop_event.set()
