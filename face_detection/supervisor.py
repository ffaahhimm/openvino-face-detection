"""Supervises the out-of-process inference worker.

This is the parent-process half of the process-isolation mode. It plays the same
pipeline role as :class:`~pipeline.InferenceWorker` (pull the latest frame from
the :class:`~camera.FrameStore`, produce detections, publish ``on_result``) but
runs the inference in a child process and adds fault recovery:

* **crash** -- if the worker process dies, it is respawned;
* **hang**  -- if a submitted frame yields no result within ``frame_timeout``,
  the worker is killed and respawned.

After ``max_restarts`` it gives up and emits ``on_fatal`` so the application can
shut down gracefully.

The process backend is injected via ``runner_factory`` so the supervisor's logic
is unit-testable with a fake backend (no real processes).
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from dataclasses import replace
from typing import Callable, Dict, Optional

from .camera import FrameStore
from .config import FallbackConfig, InferenceConfig
from .events import Event
from .fallback import FallbackPolicy, overload_concerns_device
from .logger import get_logger
from .pipeline import PipelineStats
from .worker import WorkerParams, run_device_probe, run_worker

_log = get_logger(__name__)

_UNKNOWN_DEVICE = "starting…"


def _probe_fresh_process(timeout_sec: float = 20.0) -> list:
    """Enumerate OpenVINO devices in a freshly spawned process.

    See :func:`~worker.run_device_probe` for why a fresh process is the only
    trustworthy way to check whether a previously-vanished device is back.
    """

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=run_device_probe, args=(result_q,), daemon=True)
    proc.start()
    try:
        return result_q.get(timeout=timeout_sec)
    except queue.Empty:
        return []
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2.0)


class _MPRunner:
    """Real multiprocessing backend: a spawned worker process plus its queues."""

    def __init__(self, params: WorkerParams) -> None:
        self._params = params
        self._ctx = mp.get_context("spawn")
        self._proc: Optional[mp.process.BaseProcess] = None
        self._req_q: Optional[mp.Queue] = None
        self._res_q: Optional[mp.Queue] = None

    def spawn(self) -> None:
        self._req_q = self._ctx.Queue(maxsize=2)
        self._res_q = self._ctx.Queue(maxsize=4)
        self._proc = self._ctx.Process(
            target=run_worker,
            args=(self._req_q, self._res_q, self._params),
            name="InferenceWorkerProcess",
            daemon=True,
        )
        self._proc.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def wait_ready(self, timeout: float):
        return self._res_q.get(timeout=timeout)  # raises queue.Empty

    def submit(self, frame) -> None:
        self._req_q.put_nowait(frame)  # raises queue.Full

    def poll(self):
        return self._res_q.get_nowait()  # raises queue.Empty

    def terminate(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
        self._close_queues()

    def _close_queues(self) -> None:
        # cancel_join_thread() is essential: without it the parent blocks on exit
        # trying to flush a queue whose consumer (the child) is gone.
        for q in (self._req_q, self._res_q):
            if q is not None:
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:  # noqa: BLE001
                    pass
        self._req_q = None
        self._res_q = None

    def stop(self) -> None:
        try:
            if self._req_q is not None:
                self._req_q.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        self.terminate()


RunnerFactory = Callable[[WorkerParams], "_MPRunner"]


class ProcessInferenceSupervisor(threading.Thread):
    """Drives an out-of-process worker and recovers it from crashes/hangs."""

    def __init__(
        self,
        store: FrameStore,
        params: WorkerParams,
        config: InferenceConfig,
        runner_factory: Optional[RunnerFactory] = None,
        fallback_config: Optional[FallbackConfig] = None,
    ) -> None:
        super().__init__(name="InferenceSupervisor", daemon=True)
        self._store = store
        self._config = config
        self._fallback_config = fallback_config
        # Kept mutable: after a crash on an accelerator, respawn params exclude
        # that device family so the fresh worker comes up on a working device
        # (ultimately CPU) instead of crash-looping on the broken one.
        self._params = params
        self._runner_factory: RunnerFactory = runner_factory or _MPRunner

        self._runner: Optional[_MPRunner] = None
        self._stop_event = threading.Event()
        self._exec_device = _UNKNOWN_DEVICE
        self._pending = False
        self._sent_at = 0.0
        self._inflight = None  # the frame currently being inferred (for display)
        self._restarts = 0
        # When the current worker reported ready; a worker that stayed healthy
        # long enough earns the restart budget back (crash-loop guard only).
        self._worker_ready_at = 0.0
        # Overload-driven runtime switching (mirrors the thread-mode fallback).
        # The policy survives worker respawns so quarantines are not forgotten;
        # it is only rebuilt when the usable-device set actually changes.
        self._policy: Optional[FallbackPolicy] = None
        self._policy_usable: list = []
        self._switch_lock = threading.Lock()
        self._next_switch_allowed_at = 0.0
        # Target of the switch currently compiling inside the worker (if any).
        # The worker keeps inferring on its old device meanwhile, so this only
        # gates further switch requests -- it never pauses the stream.
        self._switch_target: Optional[str] = None
        self._switch_sent_at = 0.0
        # Per-family crash count, driving the escalating exclusion duration.
        self._crash_strikes: Dict[str, int] = {}
        # Rediscovery of a vanished device: a probe thread asks a fresh process
        # whether it is back; if so this holds the family pending a deliberate
        # worker restart (a stale worker process can never re-enumerate it).
        self._probe_thread: Optional[threading.Thread] = None
        self._rediscover_family: Optional[str] = None
        # Next time the periodic preferred-device upgrade check may run.
        self._next_upgrade_check_at = 0.0
        # "No fallback available" is logged loudly once, then suppressed until
        # the situation changes -- under a low overload threshold it would
        # otherwise repeat every backoff window for hours.
        self._no_target_logged = False
        # Independent timer for probing excluded-but-possibly-recovered devices.
        # Runs on its own flat interval, NOT gated by the escalating quarantine.
        self._next_probe_check_at = 0.0

        self.stats = PipelineStats(device_provider=lambda: self._exec_device)
        self.on_result = Event("proc_result")          # (frame, detections, stats)
        self.on_device_change = Event("proc_device")   # (requested, effective)
        self.on_fatal = Event("proc_fatal")            # (reason,)

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        _log.info("Inference supervisor started (process mode)")
        if not self._start_worker():
            self._fatal("worker failed to start")
            return

        last_frame_id = 0
        while not self._stop_event.is_set():
            if not self._runner.is_alive():  # type: ignore[union-attr]
                if not self._recover(f"worker crashed (exit code {self._exit_code()})"):
                    return
                last_frame_id = 0
                continue

            rediscover = self._take_rediscovery()
            if rediscover is not None:
                if not self._restart_to_rediscover(rediscover):
                    return
                last_frame_id = 0
                continue

            self._maybe_request_upgrade()

            self._maybe_request_upgrade()
            self._maybe_probe_excluded_devices()

            if self._pending and (time.monotonic() - self._sent_at) > self._config.frame_timeout_sec:
                _log.warning(
                    "Inference timed out after %.1fs (worker hung); restarting",
                    self._config.frame_timeout_sec,
                )
                self._runner.terminate()  # type: ignore[union-attr]
                if not self._recover("inference timeout"):
                    return
                last_frame_id = 0
                continue

            self._check_switch_timeout()

            if self._pending or self._switch_in_flight():
                try:
                    self._handle_message(self._runner.poll())  # type: ignore[union-attr]
                except queue.Empty:
                    pass

            if not self._pending:
                last_frame_id = self._submit_latest(last_frame_id)

            self._stop_event.wait(0.001)

        self._shutdown_worker()
        _log.info("Inference supervisor stopped")

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _start_worker(self) -> bool:
        self._runner = self._runner_factory(self._params)
        self._runner.spawn()
        self._pending = False
        return self._await_ready()

    def _await_ready(self) -> bool:
        try:
            message = self._runner.wait_ready(self._config.worker_ready_timeout_sec)  # type: ignore[union-attr]
        except queue.Empty:
            _log.error(
                "Worker did not report ready within %.0fs",
                self._config.worker_ready_timeout_sec,
            )
            return False
        if not message or message[0] != "ready":
            detail = message[1] if message and len(message) > 1 else "unknown error"
            _log.error("Worker start failed: %s", detail)
            return False
        # Build the switch policy over the UNION of every device the workers
        # have ever reported: a worker respawned with a family excluded reports
        # only the subset it may use, and rebuilding from that subset would
        # erase both the full rotation and the active crash quarantines --
        # making the exclusion permanent instead of timed.
        usable = list(message[2]) if len(message) > 2 else []
        with self._switch_lock:
            self._switch_target = None  # a fresh worker has no switch in flight
            self._no_target_logged = False
            if self._fallback_config is not None and usable:
                union = sorted(set(self._policy_usable) | set(usable))
                if self._policy is None or set(union) != set(self._policy_usable):
                    self._policy = FallbackPolicy(
                        priority=list(self._params.auto_priority),
                        usable_devices=union,
                        cooldown_sec=self._fallback_config.cooldown_sec,
                    )
                    self._policy_usable = union
                # Boot grace: never ask a just-(re)started worker to switch
                # right away -- start-up load transients (model compile, camera
                # open) otherwise look like overload and trigger a pointless
                # immediate migration.
                self._next_switch_allowed_at = (
                    time.monotonic() + self._fallback_config.switch_backoff_sec
                )
        self._set_device(message[1])
        self._worker_ready_at = time.monotonic()
        _log.info("Worker ready; AUTO executing on: %s", self._exec_device)
        return True

    def _recover(self, reason: str) -> bool:
        """Respawn after a crash/hang. Returns False when we should stop."""

        self.stats.record_error()
        healthy_for = time.monotonic() - self._worker_ready_at
        if self._restarts and healthy_for >= self._config.restart_budget_reset_sec:
            _log.info(
                "Worker was healthy for %.0fs; restart budget reset", healthy_for
            )
            self._restarts = 0
        self._restarts += 1
        _log.warning(
            "Restarting inference worker (%d/%d): %s",
            self._restarts,
            self._config.max_restarts,
            reason,
        )
        self._exclude_crashed_device()
        if self._runner is not None:
            self._runner.terminate()
        if self._restarts > self._config.max_restarts:
            self._fatal(f"exceeded max restarts ({self._config.max_restarts})")
            return False
        if self._stop_event.wait(self._config.restart_delay_sec):
            return False  # stop requested during backoff
        if not self._start_worker():
            self._fatal("worker failed to restart")
            return False
        _log.info("Worker recovered; AUTO executing on: %s", self._exec_device)
        return True

    def _exclude_crashed_device(self) -> None:
        """Ban the device family the worker died on -- for a while, not forever.

        A worker that crashed or hung while executing on an accelerator will
        almost certainly do so again if respawned onto it immediately: the
        family is dropped from the respawn parameters (the fresh worker comes
        up on the next healthy device, ultimately CPU) and quarantined in the
        switch policy. The ban is *timed and escalating* -- doubling per repeat
        offence -- because a single hang can just as well mean the whole
        machine was starved as that the device is broken; a wrongly-blamed GPU
        gets another chance after the cooldown, while a genuinely bad one
        drifts out of rotation for longer and longer. A later successful
        switch back fully rehabilitates it (see :meth:`_rehabilitate`). CPU is
        never excluded: it is the last resort.
        """

        family = self._exec_device.split(".", 1)[0].upper()
        if family == "GPU" and self._params.allow_gpu:
            self._params = replace(self._params, allow_gpu=False)
        elif family == "NPU" and self._params.allow_npu:
            self._params = replace(self._params, allow_npu=False)
        else:
            return  # CPU, unknown, or already excluded: nothing to do
        strikes = self._crash_strikes.get(family, 0) + 1
        self._crash_strikes[family] = strikes
        duration = self._config.crash_cooldown_sec * (2 ** min(strikes - 1, 3))
        with self._switch_lock:
            if self._policy is not None:
                self._policy.quarantine_for(family, duration)
        _log.warning(
            "Worker died while executing on %s; excluding %s for %.0fs "
            "(offence #%d) and falling back to the remaining devices",
            self._exec_device,
            family,
            duration,
            strikes,
        )
        self._exec_device = _UNKNOWN_DEVICE

    def _rehabilitate(self, device: str, why: str = "a successful runtime switch") -> None:
        family = device.split(".", 1)[0].upper()
        if family == "GPU" and not self._params.allow_gpu:
            self._params = replace(self._params, allow_gpu=True)
        elif family == "NPU" and not self._params.allow_npu:
            self._params = replace(self._params, allow_npu=True)
        else:
            return
        self._crash_strikes.pop(family, None)   # <-- add this line
        _log.info("%s rehabilitated (%s); re-enabled for worker respawns", family, why)
    # ------------------------------------------------------------------
    # Rediscovery of a vanished device
    # ------------------------------------------------------------------
    def _start_absent_probe(self, family: str) -> None:
        """Check from a fresh process whether an absent device is back.

        Runs on its own thread because the probe spawns a process and imports
        OpenVINO (several seconds); at most one probe runs at a time, and it
        is only ever triggered by an (escalatingly rare) failed switch, so a
        permanently-dead device costs a rare, quiet background check.
        """

        if self._probe_thread is not None and self._probe_thread.is_alive():
            return
        family = family.split(".", 1)[0].upper()

        def _probe() -> None:
            families = {
                d.split(".", 1)[0].upper() for d in _probe_fresh_process()
            }
            if family in families:
                _log.info(
                    "%s is visible to a fresh device probe; scheduling a worker "
                    "restart to rediscover it",
                    family,
                )
                with self._switch_lock:
                    self._rediscover_family = family
            else:
                _log.info(
                    "%s still absent system-wide (fresh probe saw: %s)",
                    family,
                    sorted(families) if families else "nothing",
                )

        self._probe_thread = threading.Thread(target=_probe, name="DeviceProbe", daemon=True)
        self._probe_thread.start()

    def _take_rediscovery(self) -> Optional[str]:
        with self._switch_lock:
            family, self._rediscover_family = self._rediscover_family, None
        return family

    def _restart_to_rediscover(self, family: str) -> bool:
        """Deliberately restart the worker so it can see a recovered device.

        The stale worker process can never re-enumerate a device that was
        absent when it started, so a short planned restart (~2s on the warm
        cache) is the only way to actually get the device back in use.
        """

        with self._switch_lock:
            if self._policy is not None:
                self._policy.clear_strikes(family)
        self._rehabilitate(family, why="a fresh probe found it present again")
        _log.info("Restarting inference worker to rediscover %s", family)
        if self._runner is not None:
            self._runner.stop()
        if self._start_worker():
            return True
        # Fall back to crash recovery so the restart budget / fatal logic apply.
        return self._recover(f"restart to rediscover {family} failed")

    def _shutdown_worker(self) -> None:
        if self._runner is not None:
            self._runner.stop()

    # ------------------------------------------------------------------
    # Frame flow
    # ------------------------------------------------------------------
    def _submit_latest(self, last_frame_id: int) -> int:
        captured = self._store.get()
        if captured is None or captured.frame_id == last_frame_id:
            return last_frame_id
        try:
            self._runner.submit(captured.image)  # type: ignore[union-attr]
        except queue.Full:
            return last_frame_id
        self._inflight = captured.image
        self._pending = True
        self._sent_at = time.monotonic()
        return captured.frame_id

    def _handle_message(self, message) -> None:
        kind = message[0]
        if kind == "result":
            _, detections, inference_ms, device = message
            self.stats.record_success(inference_ms, len(detections))
            self._set_device(device)
            self.on_result.emit(self._inflight, detections, self.stats.snapshot())
            self._pending = False
        elif kind == "switched":
            with self._switch_lock:
                self._switch_target = None
                if self._policy is not None:
                    # The device proved itself: forget its failure escalation.
                    self._policy.clear_strikes(message[1])
            self._set_device(message[1])
            self._rehabilitate(message[1])
            _log.info("Worker switched; now executing on: %s", self._exec_device)
        elif kind == "switch_failed":
            target, detail = message[1], message[2]
            # Native plugin errors span many lines; keep the log to the point.
            first_line = detail.splitlines()[0] if detail else detail
            _log.warning("Worker switch to '%s' failed: %s", target, first_line)
            _log.debug("Full switch failure detail for '%s': %s", target, detail)
            with self._switch_lock:
                self._switch_target = None
                if self._policy is not None:
                    # Escalating: a target that keeps refusing (e.g. a GPU whose
                    # driver aborted and left no adapter behind) is probed less
                    # and less often -- yet still rehabilitates automatically
                    # if the device ever comes back and a switch succeeds.
                    self._policy.quarantine(target, escalate=True)
            if "not currently present" in detail:
                # The live worker cannot see the device -- but its process may
                # simply be stale (enumeration is cached per process). Ask a
                # fresh process whether the device is back at the system level.
                self._start_absent_probe(target)
        elif kind == "error":
            _log.warning("Worker error: %s", message[1])
            self._runner.terminate()  # type: ignore[union-attr]
            self._pending = False  # alive-check next loop triggers recovery

    # ------------------------------------------------------------------
    # Overload-driven runtime switching (called from the monitor thread)
    # ------------------------------------------------------------------
    def handle_overload(self, detail: str) -> None:
        """Ask the worker to migrate off an overloaded device at runtime.

        Mirrors the thread-mode :class:`~fallback.FallbackController`: the
        switch only happens when the saturated resource is the one running
        inference (see :func:`~fallback.overload_concerns_device`), attempts
        are rate-limited by the switch backoff, and the device being fled is
        quarantined so the next overload cannot bounce straight back. The
        migration itself runs inside the worker via a ``("switch", device)``
        control message -- the parent process never touches the accelerator.
        """

        if self._fallback_config is None or not self._fallback_config.switch_on_overload:
            return
        with self._switch_lock:
            policy, runner = self._policy, self._runner
            if policy is None or runner is None or self._stop_event.is_set():
                return
            if self._switch_target is not None:
                return  # a switch is already compiling inside the worker
            family = self._exec_device.split(".", 1)[0].upper()
            if family not in ("CPU", "GPU", "NPU"):
                return  # worker still starting; nothing to switch yet
            if not overload_concerns_device(detail, family):
                _log.debug(
                    "Overload (%s) does not concern worker device '%s'; staying",
                    detail,
                    self._exec_device,
                )
                return
            now = time.monotonic()
            if now < self._next_switch_allowed_at:
                return
            self._next_switch_allowed_at = now + self._fallback_config.switch_backoff_sec
            target = policy.next_device(family, now=now)
            if target is None:
                if not self._no_target_logged:
                    self._no_target_logged = True
                    _log.error(
                        "No overload fallback device available (%s); will keep "
                        "checking quietly and switch when one frees up",
                        detail,
                    )
                else:
                    _log.debug("Still no overload fallback device (%s)", detail)
                return
            self._no_target_logged = False
            try:
                runner.submit(("switch", target))
            except Exception:  # noqa: BLE001 - queue full/closed: retry on next signal
                return
            self._switch_target = target
            self._switch_sent_at = now
            policy.quarantine(family, now=now)
            _log.warning(
                "Overload detected (%s); asking worker to switch %s -> %s",
                detail,
                self._exec_device,
                target,
            )

    def _maybe_probe_excluded_devices(self) -> None:
        """Independently probe whether a crashed/excluded device has come back.

        This runs on a short, FLAT interval (20 s) regardless of how many times
        the device has failed -- so escalating crash-quarantines do NOT delay
        rediscovery.  The probe itself is cheap: it just spawns a tiny process
        to ask OpenVINO 'what devices can you see?'  A successful probe schedules
        a worker restart (_restart_to_rediscover) which is the only way to pick
        the device back up once the live worker's ov.Core() cache has gone stale.
        """
        now = time.monotonic()
        if now < self._next_probe_check_at:
            return
        # Flat 20-second interval -- intentionally not escalating.
        self._next_probe_check_at = now + 20.0

        # Find any device family that is currently excluded from params
        # (allow_gpu=False or allow_npu=False) but is NOT already being probed.
        if self._probe_thread is not None and self._probe_thread.is_alive():
            return  # a probe is already running; don't stack them

        excluded = []
        if not self._params.allow_gpu:
            excluded.append("GPU")
        if not self._params.allow_npu:
            excluded.append("NPU")

        if not excluded:
            return  # nothing excluded, nothing to probe

        # Probe for the first excluded family (GPU takes priority over NPU).
        self._start_absent_probe(excluded[0])

    def _maybe_request_upgrade(self) -> None:        """Move inference back onto a recovered higher-priority device.

        Runs on the supervisor loop, self-rate-limited to
        ``preferred_retry_sec``. Deliberately independent of overload signals:
        a GPU that crashed and later recovered gets the work back even when
        the CPU is coping fine. The request goes through the exact same
        machinery as an overload switch, so a still-broken device answers
        with a clean refusal, an escalating quarantine, and -- if it is only
        the worker's stale enumeration -- the fresh-process probe and a
        rediscovery restart.
        """

        if self._fallback_config is None or self._fallback_config.preferred_retry_sec <= 0:
            return
        with self._switch_lock:
            policy, runner = self._policy, self._runner
            if policy is None or runner is None or self._switch_target is not None:
                return
            now = time.monotonic()
            if now < self._next_upgrade_check_at:
                return
            self._next_upgrade_check_at = now + self._fallback_config.preferred_retry_sec
            family = self._exec_device.split(".", 1)[0].upper()
            if family not in ("CPU", "GPU", "NPU"):
                return
            target = policy.preferred_upgrade(family, now=now)
            if target is None:
                return
            if now < self._next_switch_allowed_at:
                return
            self._next_switch_allowed_at = now + self._fallback_config.switch_backoff_sec
            try:
                runner.submit(("switch", target))
            except Exception:  # noqa: BLE001 - queue full/closed: retry next period
                return
            self._switch_target = target
            self._switch_sent_at = now
            _log.info(
                "Preferred device '%s' may be available again; asking worker to "
                "switch back from %s",
                target,
                self._exec_device,
            )

    def _switch_in_flight(self) -> bool:
        with self._switch_lock:
            return self._switch_target is not None

    def _check_switch_timeout(self) -> None:
        """Abandon a runtime switch that never produced a reply.

        The worker keeps inferring on its old device during a background
        switch-compile, so a wedged compile is NOT fatal and must not restart
        anything -- but the in-flight marker has to be cleared (or overload
        switching would be blocked forever) and the unresponsive target
        quarantined.
        """

        with self._switch_lock:
            target = self._switch_target
            if target is None:
                return
            timeout = self._config.worker_ready_timeout_sec
            if time.monotonic() - self._switch_sent_at <= timeout:
                return
            self._switch_target = None
            if self._policy is not None:
                self._policy.quarantine(target)
        _log.warning(
            "Runtime switch to '%s' produced no reply within %.0fs; abandoning it "
            "and staying on the current device",
            target,
            timeout,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_device(self, device: str) -> None:
        if not device:
            return
        with self._switch_lock:
            changed = device != self._exec_device
            if changed:
                self._exec_device = device
        if changed:
            self.on_device_change.emit("AUTO", device)

    def _exit_code(self):
        proc = getattr(self._runner, "_proc", None)
        return getattr(proc, "exitcode", None)

    def _fatal(self, reason: str) -> None:
        _log.error("Inference worker unrecoverable: %s", reason)
        self.on_fatal.emit(reason)

    def stop(self) -> None:
        self._stop_event.set()
