"""Child-process inference worker (opt-in process-isolation mode).

Runs the OpenVINO inference loop in a *separate process* so that a device hang
or crash (driver fault, GPU reset, segfault) is contained: the supervisor in the
parent process detects it and respawns this worker. The actual inference reuses
:class:`~device_manager.DeviceManager` and :class:`~inference.FaceDetector`, so
there is no duplicated model logic.

:func:`run_worker` must stay a top-level, picklable function: the ``spawn`` start
method (default on Windows/macOS) re-imports this module and re-invokes it in the
child. Everything passed to it -- the queues and :class:`WorkerParams` -- must be
picklable.

Protocol (messages the worker puts on ``result_q``):
    ("ready",  device_str, usable_devices)       once, after compiling
    ("result", detections, inference_ms, device) per processed frame
    ("switched", device_str)                     after a runtime device switch
    ("switch_failed", target, message)           switch failed; old device kept
    ("error",  message)                          on failure, then the worker exits
The supervisor puts raw BGR frames (``np.ndarray``) on ``request_q``; a
``("switch", device)`` tuple asks the worker to migrate inference onto another
device at runtime (overload-driven); ``None`` tells it to exit cleanly.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
from dataclasses import dataclass
from typing import Tuple

from .config import DeviceConfig, LoggingConfig
from .device_manager import DeviceManager
from .inference import FaceDetector
from .logger import get_logger, setup_logging

_log = get_logger(__name__)


@dataclass(frozen=True)
class WorkerParams:
    """Everything the worker needs to build the detector (must be picklable)."""

    model_xml: str
    device: str  # "AUTO", "AUTO:GPU,CPU", "CPU", "GPU.0", ...
    performance_hint: str
    confidence_threshold: float
    cache_dir: str
    auto_priority: Tuple[str, ...]
    allow_gpu: bool
    allow_npu: bool


def run_device_probe(result_q: "mp.Queue") -> None:
    """Report the devices a FRESH process can see (spawned by the supervisor).

    GPU runtimes cache their device enumeration per process: a worker spawned
    while the driver was down keeps answering "not present" even after the
    driver recovers. Only a brand-new process re-enumerates from scratch, so
    this tiny probe is the reliable way to ask "is the device back?" at the
    system level.
    """

    _silence_crash_dialogs()
    try:
        import openvino as ov

        result_q.put(list(ov.Core().available_devices))
    except Exception:  # noqa: BLE001 - a broken runtime means "saw nothing"
        result_q.put([])


def _silence_crash_dialogs() -> None:
    """Stop Windows from popping a modal dialog when the worker faults, so a
    crash exits the process cleanly for the supervisor to observe."""

    if sys.platform == "win32":
        try:
            import ctypes

            # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
        except Exception:  # noqa: BLE001
            pass


def _build_detector(params: WorkerParams) -> Tuple[FaceDetector, DeviceManager]:
    device_config = DeviceConfig(
        auto_priority=list(params.auto_priority),
        performance_hint=params.performance_hint,
        allow_gpu=params.allow_gpu,
        allow_npu=params.allow_npu,
        cache_dir=params.cache_dir,
    )
    manager = DeviceManager(device_config)
    manager.load_model(params.model_xml)
    if params.device == "AUTO":
        compiled = manager.compile_auto()
    else:
        compiled = manager.compile_for(params.device)
    return FaceDetector(compiled, params.confidence_threshold), manager


def run_worker(request_q: "mp.Queue", result_q: "mp.Queue", params: WorkerParams) -> None:
    """Entry point executed in the child process."""

    _silence_crash_dialogs()
    # The spawned child starts with an unconfigured root logger, so anything it
    # logs would fall through to the bare last-resort stderr handler. Configure
    # console logging (same format as the parent, which shares this console).
    setup_logging(LoggingConfig(level=os.environ.get("FD_LOG_LEVEL", "INFO")))

    try:
        detector, manager = _build_detector(params)
    except Exception as exc:  # noqa: BLE001 - report and exit; supervisor decides
        result_q.put(("error", f"worker init failed: {exc}"))
        return

    result_q.put(("ready", manager.effective_device(), manager.usable_devices()))
    _log.info("Worker ready on %s", manager.effective_device())

    switch_busy = threading.Event()
    while True:
        item = request_q.get()
        if item is None:  # graceful stop from the supervisor
            return
        if isinstance(item, tuple) and item and item[0] == "switch":
            _start_switch(item[1], detector, manager, result_q, switch_busy)
            continue
        try:
            result = detector.infer(item)
        except Exception as exc:  # noqa: BLE001 - exit so a fresh worker is spawned
            result_q.put(("error", f"inference failed: {exc}"))
            return
        result_q.put(
            ("result", result.detections, result.inference_ms, manager.effective_device())
        )


def _start_switch(
    target: str,
    detector: FaceDetector,
    manager: DeviceManager,
    result_q: "mp.Queue",
    busy: threading.Event,
) -> None:
    """Begin migrating inference onto ``target`` on a background thread.

    The compile can take many seconds (cold kernel cache, loaded machine), so
    it must NOT run in the message loop: frames keep flowing on the current
    device the whole time, the stream never stalls, and the supervisor's hang
    detector sees a perfectly responsive worker. Only when the new model is
    ready is it atomically swapped in. A failed switch keeps the current
    model -- a degraded-but-working device beats no inference at all -- and is
    reported so the supervisor can quarantine the target.
    """

    if busy.is_set():
        result_q.put(("switch_failed", target, "another switch is already in progress"))
        return
    busy.set()

    def _compile_and_swap() -> None:
        try:
            # A device can vanish at runtime -- e.g. the GPU driver aborted
            # earlier and the adapter is gone until it recovers. Re-query the
            # runtime instead of compiling blind: a missing device then costs
            # one clean line, not a multi-line native plugin exception.
            try:
                present = {d.split(".", 1)[0].upper() for d in manager.available_devices()}
            except Exception:  # noqa: BLE001 - if even the query fails, let compile decide
                present = None
            family = target.split(".", 1)[0].upper()
            if present is not None and family not in present:
                _log.warning(
                    "Runtime switch to '%s' refused: device not currently present "
                    "(available: %s)", target, sorted(present),
                )
                result_q.put(("switch_failed", target, "device not currently present"))
                return
            try:
                compiled = manager.switch_device(target)
                detector.set_compiled_model(compiled)
            except Exception as exc:  # noqa: BLE001 - keep running on the old device
                _log.warning(
                    "Runtime switch to '%s' failed: %s",
                    target,
                    str(exc).splitlines()[0] if str(exc) else exc,
                )
                result_q.put(("switch_failed", target, str(exc)))
                return
            _log.info("Runtime switch complete; now on %s", manager.effective_device())
            result_q.put(("switched", manager.effective_device()))
        finally:
            busy.clear()

    threading.Thread(target=_compile_and_swap, name="WorkerSwitch", daemon=True).start()
