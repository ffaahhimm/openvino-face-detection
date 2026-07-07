"""Application coordinator.

:class:`Application` is the composition root: it constructs every component,
wires their events together, owns the thread lifecycle, and exposes a small
public API. It contains *coordination* logic only -- no face detection, no
OpenVINO calls, no camera I/O; those live in their dedicated modules.

Two inference back-ends share the same pipeline, chosen by
``config.inference.mode``:

* ``thread``  -- in-process :class:`~pipeline.InferenceWorker` + in-process
  device fallback (lowest latency, the default);
* ``process`` -- an isolated worker process driven by
  :class:`~supervisor.ProcessInferenceSupervisor`, which survives device hangs
  and crashes by respawning the worker.

Both expose the same ``on_result`` event and a :class:`~pipeline.PipelineStats`,
so the camera, monitor, reporter and GUI are identical in either mode.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

from .camera import Camera, CameraThread, FrameStore
from .config import AppConfig
from .device_manager import DeviceManager, DeviceError
from .events import Event
from .fallback import FallbackController, FallbackPolicy
from .inference import FaceDetector
from .logger import get_logger
from .monitor import ResourceMonitor
from .pipeline import InferenceWorker, PipelineStats
from .reporting import ConsoleReporter

_log = get_logger(__name__)


class ApplicationError(RuntimeError):
    """Raised when the application cannot be initialised."""


class Application:
    """Owns and orchestrates the whole face-detection backend."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._initialised = False
        self._shutdown_event = threading.Event()
        # Guards teardown so it runs exactly once and every caller blocks until
        # it is fully complete (see _teardown).
        self._teardown_lock = threading.Lock()
        self._torn_down = False

        # Components (created in initialize()).
        self._camera: Optional[Camera] = None
        self._frame_store: Optional[FrameStore] = None
        self._camera_thread: Optional[CameraThread] = None
        self._monitor: Optional[ResourceMonitor] = None
        self._reporter: Optional[ConsoleReporter] = None
        self._stats: Optional[PipelineStats] = None
        # The inference service is a Thread with .start()/.stop()/.on_result --
        # either an InferenceWorker (thread mode) or a supervisor (process mode).
        self._inference: Optional[threading.Thread] = None
        # Thread-mode-only components.
        self._device_manager: Optional[DeviceManager] = None
        self._detector: Optional[FaceDetector] = None
        self._fallback: Optional[FallbackController] = None

        # Public events -- the GUI-ready integration surface.
        self.on_result = Event("app_result")            # (frame, detections, stats)
        self.on_device_change = Event("app_device")     # (requested, effective)
        self.on_resource = Event("app_resource")        # (ResourceSnapshot,)
        self.on_fallback = Event("app_fallback")        # (from_device, to_device)

    # ------------------------------------------------------------------
    # Initialisation / wiring
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Build every component and connect their events. Idempotent."""

        if self._initialised:
            return
        mode = self._config.inference.mode
        from . import __version__

        _log.info("Initialising application (build %s, %s inference mode)", __version__, mode)

        self._validate_model()
        # The two slowest start-up steps -- opening the camera (device
        # enumeration) and preparing the inference engine (model compile) -- are
        # independent, so run them concurrently to roughly halve start-up time.
        self._init_camera_and_inference(mode)
        # Give the monitor a view of inference latency so it can flag a device
        # that is overloaded/degraded (slow), not just high system CPU/RAM.
        self._monitor = ResourceMonitor(
            self._config.monitor,
            latency_provider=lambda: self._stats.snapshot().avg_inference_ms,  # type: ignore[union-attr]
        )

        self._reporter = ConsoleReporter(
            stats=self._stats,          # type: ignore[arg-type]
            monitor=self._monitor,
            interval_sec=self._config.report_interval_sec,
        )
        self._wire_events()
        self._initialised = True
        _log.info("Initialisation complete")

    def _validate_model(self) -> None:
        xml = self._config.model.xml_path
        if not os.path.exists(xml):
            raise ApplicationError(
                f"Model file not found: '{xml}'. "
                "Download it first (see README / scripts/download_models.py)."
            )

    def _init_camera_and_inference(self, mode: str) -> None:
        # The frame store is shared, cheap, and needed by the inference engine,
        # so create it up front; only the (slow) camera open runs on the thread.
        self._frame_store = FrameStore()
        errors: dict = {}

        def open_camera() -> None:
            try:
                self._open_camera()
            except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
                errors["camera"] = exc

        cam_thread = threading.Thread(target=open_camera, name="CameraOpen", daemon=True)
        cam_thread.start()
        try:
            if mode == "process":
                self._init_process_inference()
            else:
                self._init_thread_inference()
        finally:
            cam_thread.join()

        if "camera" in errors:
            exc = errors["camera"]
            if isinstance(exc, ApplicationError):
                raise exc
            raise ApplicationError(f"Failed to open camera: {exc}") from exc

    def _open_camera(self) -> None:
        camera_cfg = self._config.camera
        camera = Camera(camera_cfg)
        try:
            camera.open()
        except Exception as exc:  # noqa: BLE001 - surface a clean init error
            raise ApplicationError(f"Failed to open camera: {exc}") from exc
        self._camera = camera
        self._camera_thread = CameraThread(camera, self._frame_store, camera_cfg)

    def _init_thread_inference(self) -> None:
        """In-process detector + AUTO device fallback (default mode)."""

        model_cfg = self._config.model
        device_manager = DeviceManager(self._config.device)
        try:
            device_manager.load_model(model_cfg.xml_path, model_cfg.bin_path)
            compiled = device_manager.compile_auto()
        except DeviceError as exc:
            raise ApplicationError(f"Failed to prepare inference engine: {exc}") from exc

        self._device_manager = device_manager
        self._detector = FaceDetector(compiled, model_cfg.confidence_threshold)

        # Warm a CPU standby in the background so a failing accelerator can be
        # answered with an instant switch instead of an on-crisis recompile.
        usable = device_manager.usable_devices()
        if "CPU" in usable and len(usable) > 1:
            threading.Thread(
                target=lambda: device_manager.prepare_standby("CPU"),
                name="StandbyCompile",
                daemon=True,
            ).start()

        policy = FallbackPolicy(
            priority=self._config.device.auto_priority,
            usable_devices=device_manager.usable_devices(),
            cooldown_sec=self._config.fallback.cooldown_sec,
        )
        self._fallback = FallbackController(
            config=self._config.fallback,
            device_manager=device_manager,
            detector=self._detector,
            policy=policy,
        )
        self._stats = PipelineStats(device_provider=device_manager.effective_device)
        self._inference = InferenceWorker(
            store=self._frame_store,        # type: ignore[arg-type]
            detector=self._detector,
            stats=self._stats,
            fallback=self._fallback,
        )
        _log.info("Usable devices: %s", device_manager.usable_devices())

    def _init_process_inference(self) -> None:
        """Isolated worker process supervised for crash/hang recovery."""

        # Imported lazily so thread mode never pulls in the multiprocessing stack.
        from .supervisor import ProcessInferenceSupervisor
        from .worker import WorkerParams

        model_cfg = self._config.model
        device_cfg = self._config.device
        params = WorkerParams(
            model_xml=os.path.abspath(model_cfg.xml_path),
            device="AUTO",
            performance_hint=device_cfg.performance_hint,
            confidence_threshold=model_cfg.confidence_threshold,
            cache_dir=device_cfg.cache_dir,
            auto_priority=tuple(device_cfg.auto_priority),
            allow_gpu=device_cfg.allow_gpu,
            allow_npu=device_cfg.allow_npu,
        )
        supervisor = ProcessInferenceSupervisor(
            self._frame_store,              # type: ignore[arg-type]
            params,
            self._config.inference,
            fallback_config=self._config.fallback,
        )
        self._inference = supervisor
        self._stats = supervisor.stats

    def _wire_events(self) -> None:
        """Connect internal events and re-publish them on the public surface."""

        assert self._inference is not None and self._monitor is not None

        self._inference.on_result.subscribe(self.on_result.emit)  # type: ignore[attr-defined]
        self._monitor.on_sample.subscribe(self.on_resource.emit)

        if self._config.inference.mode == "process":
            self._inference.on_device_change.subscribe(self.on_device_change.emit)  # type: ignore[attr-defined]
            # Overload signals drive a runtime device switch inside the worker.
            self._monitor.on_overload.subscribe(self._inference.handle_overload)  # type: ignore[attr-defined]
            # An unrecoverable worker asks the app to shut down cleanly.
            self._inference.on_fatal.subscribe(self._on_inference_fatal)  # type: ignore[attr-defined]
        else:
            assert self._device_manager is not None and self._fallback is not None
            self._monitor.on_overload.subscribe(self._fallback.handle_overload)
            # Per-second samples drive the (self-rate-limited) check that moves
            # inference back onto a recovered higher-priority device.
            self._monitor.on_sample.subscribe(self._fallback.maybe_upgrade)
            self._device_manager.device_changed.subscribe(self.on_device_change.emit)
            self._fallback.fallback_occurred.subscribe(self.on_fallback.emit)

    def _on_inference_fatal(self, reason: str) -> None:
        _log.error("Inference unrecoverable (%s); shutting down", reason)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start all threads and block until :meth:`shutdown` is requested.

        Teardown always happens here, in the ``finally`` block, so the pipeline
        is fully stopped before ``run`` returns -- regardless of whether the stop
        was triggered by :meth:`shutdown` (from any thread) or a
        ``KeyboardInterrupt``.
        """

        # Initialise and start under the teardown lock so a shutdown() from
        # another thread can never interleave with start-up; it simply waits,
        # then tears down cleanly. A startup failure still propagates out.
        with self._teardown_lock:
            if self._torn_down:
                _log.info("Shutdown requested before start-up; pipeline not started")
                return
            self.initialize()
            _log.info("Starting threads")
            # Start producers before consumers.
            self._camera_thread.start()  # type: ignore[union-attr]
            self._monitor.start()        # type: ignore[union-attr]
            self._inference.start()      # type: ignore[union-attr]
            self._reporter.start()       # type: ignore[union-attr]
            _log.info("Application running. Press Ctrl-C to stop.")

        try:
            self._shutdown_event.wait()
        finally:
            self._teardown()

    def shutdown(self) -> None:
        """Request shutdown from any thread (e.g. a GUI or a signal handler)."""

        self._shutdown_event.set()
        self._teardown()

    def _teardown(self) -> None:
        """Stop every thread in reverse dependency order and release the camera.

        Runs at most once; bounded, and safe against a Ctrl-C landing mid-way.
        """

        with self._teardown_lock:
            if self._torn_down:
                return
            self._torn_down = True  # set first: a Ctrl-C mid-teardown won't re-run it
            _log.info("Shutting down")

            # Consumers first, then producers, so nothing reads a released device.
            threads = (self._reporter, self._inference, self._monitor, self._camera_thread)
            for thread in threads:
                if thread is not None:
                    thread.stop()  # type: ignore[attr-defined]

            # Bounded joins. A thread wedged in a blocking OpenCV/OS call is a
            # daemon and will be reaped on process exit; a Ctrl-C is swallowed so
            # teardown always completes.
            for thread in threads:
                if thread is not None and thread.is_alive():
                    try:
                        thread.join(timeout=2.0)
                    except KeyboardInterrupt:
                        pass
                    if thread.is_alive():
                        _log.warning(
                            "%s did not stop in time; it is a daemon and will exit "
                            "with the process", thread.name,
                        )

            # Only release the camera if its thread actually stopped, to avoid a
            # release/read race with a still-running capture thread.
            camera_stopped = (
                self._camera_thread is None or not self._camera_thread.is_alive()
            )
            if self._camera is not None and camera_stopped:
                self._camera.release()
            _log.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Public API (usable directly or by a UI / CLI)
    # ------------------------------------------------------------------
    def available_devices(self) -> List[str]:
        """Devices usable for inference (empty in process mode -- AUTO decides)."""

        if self._device_manager is not None:
            return self._device_manager.usable_devices()
        return []

    def switch_device(self, device: str) -> bool:
        """Request a runtime device switch (thread mode only)."""

        if self._fallback is None:
            _log.warning(
                "Runtime device switching is unavailable in process mode; the "
                "AUTO plugin selects the device."
            )
            return False
        return self._fallback.switch_to(device)

    @property
    def config(self) -> AppConfig:
        return self._config
