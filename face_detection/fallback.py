"""Application-level device recovery and fallback.

The OpenVINO AUTO plugin already handles *load-time* fallback (picking an
available device when the model is first compiled). This module adds *run-time*
resilience: when the active device starts failing or the machine is overloaded,
we deliberately migrate inference to another device.

Two collaborators:

* :class:`FallbackPolicy`     -- pure decision logic: given the priority order,
  the set of usable devices, and which devices are temporarily quarantined,
  decide what to switch to. No side effects, trivially unit-testable.
* :class:`FallbackController` -- orchestration: counts failures, asks the policy
  for a target, drives :class:`~device_manager.DeviceManager` to recompile, and
  rebinds the :class:`~inference.FaceDetector`.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from .config import FallbackConfig
from .device_manager import DeviceManager, DeviceError
from .events import Event
from .inference import FaceDetector
from .logger import get_logger

_log = get_logger(__name__)


def _family(device_name: str) -> str:
    """Base family for a device id or AUTO string (``"AUTO:GPU,CPU"`` -> ``"GPU"``)."""

    name = device_name.upper()
    if name.startswith("AUTO"):
        # Prefer the first concrete candidate if present, else just "AUTO".
        parts = name.split(":", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1].split(",", 1)[0].split(".", 1)[0]
        return "AUTO"
    return name.split(".", 1)[0]


# Error-message markers indicating the device itself died (driver reset, context
# lost, out of device memory, ...) rather than a transient per-frame problem.
# Such a device will not heal between frames, so we flee it on the *first* error
# instead of waiting for the consecutive-failure threshold.
_FATAL_ERROR_MARKERS = (
    "device_lost",
    "device lost",
    "ze_result_error",
    "cl_out_of",
    "clenqueue",
    "clwaitforevents",
    "out_of_device_memory",
    "gpu hang",
    "device is lost",
    "context lost",
    "device removed",
)


def _looks_fatal(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _FATAL_ERROR_MARKERS)


def overload_concerns_device(detail: str, device_family: str) -> bool:
    """Decide whether an overload signal justifies moving inference off
    ``device_family``.

    The monitor's detail string lists breached resources ("CPU 93%, MEM 91%").
    Migrating inference only helps when the *saturated* resource is the one
    doing the inference:

    * ``LATENCY`` -- the active device itself is struggling: always move.
    * ``CPU`` / ``GPU`` -- move only if inference currently runs there;
      fleeing a GPU because the *CPU* is busy lands the work on the busy CPU.
    * ``MEM`` -- system-wide; switching devices does not free RAM, so a memory
      breach alone never triggers a switch (this was the source of observed
      CPU<->GPU thrash under sustained memory pressure).
    """

    family = device_family.upper()
    for part in detail.split(","):
        resource = part.strip().split(" ", 1)[0].upper()
        if resource == "LATENCY":
            return True
        if resource in ("CPU", "GPU", "NPU") and resource == family:
            return True
    return False


class FallbackPolicy:
    """Chooses the next device, honouring priority order and cooldowns.

    A device that recently failed is *quarantined* until its cooldown expires so
    we do not immediately bounce back onto a broken accelerator. CPU is treated
    as the guaranteed last resort and is never left without a candidate.
    """

    def __init__(
        self,
        priority: List[str],
        usable_devices: List[str],
        cooldown_sec: float,
    ) -> None:
        self._priority = [d.upper() for d in priority]
        self._usable = [d.upper() for d in usable_devices]
        self._cooldown_sec = cooldown_sec
        # family -> monotonic timestamp until which it stays quarantined.
        self._quarantine: Dict[str, float] = {}
        # family -> consecutive error-quarantines; drives escalating cooldowns.
        self._strikes: Dict[str, int] = {}

    def _ordered_candidates(self) -> List[str]:
        """Usable devices sorted by configured priority (unknowns go last)."""

        def rank(device: str) -> int:
            return self._priority.index(device) if device in self._priority else len(self._priority)

        return sorted(self._usable, key=rank)

    def quarantine(
        self, device: str, now: Optional[float] = None, escalate: bool = False
    ) -> None:
        """Mark ``device``'s family as temporarily unusable.

        With ``escalate=True`` (used for *error* quarantines, not overload) the
        cooldown doubles on every repeat offence, so a device that keeps failing
        drifts out of rotation instead of being retried every base cooldown --
        e.g. an unstable GPU cannot keep pulling inference back onto itself.
        """

        now = time.monotonic() if now is None else now
        family = _family(device)
        if escalate:
            strikes = self._strikes.get(family, 0) + 1
            self._strikes[family] = strikes
            cooldown = self._cooldown_sec * (2 ** min(strikes - 1, 4))
        else:
            cooldown = self._cooldown_sec
        self._quarantine[family] = now + cooldown
        _log.info("Device '%s' quarantined for %.0fs", family, cooldown)

    def clear_strikes(self, device: str) -> None:
        """Forget escalation history for ``device`` (it proved healthy again)."""

        self._strikes.pop(_family(device), None)

    def quarantine_for(self, device: str, duration_sec: float, now: Optional[float] = None) -> None:
        """Quarantine ``device`` for an explicit duration.

        Used for crash exclusions, whose (long, escalating) duration is decided
        by the caller rather than by the policy's own cooldown.
        """

        now = time.monotonic() if now is None else now
        family = _family(device)
        self._quarantine[family] = now + duration_sec
        _log.info("Device '%s' quarantined for %.0fs", family, duration_sec)

    def remove_device(self, device: str) -> None:
        """Permanently drop ``device``'s family from the usable set.

        Used when a device family is banned for the session (e.g. the process
        supervisor excluded it after a native crash) -- it must never be
        offered as a switch target again, not even as a last resort.
        """

        family = _family(device)
        if family in self._usable:
            self._usable.remove(family)
            _log.info("Device '%s' removed from fallback rotation", family)

    def _is_quarantined(self, device: str, now: float) -> bool:
        expiry = self._quarantine.get(device)
        if expiry is None:
            return False
        if now >= expiry:
            # Cooldown elapsed -- the device is eligible again.
            del self._quarantine[device]
            return False
        return True

    def next_device(
        self, current: str, now: Optional[float] = None
    ) -> Optional[str]:
        """Return the best device to switch to, or ``None`` if none is available.

        Skips the currently-active family and any quarantined family. CPU is
        offered even if quarantined when nothing else remains, because running on
        a busy CPU still beats stopping inference entirely.
        """

        now = time.monotonic() if now is None else now
        current_family = _family(current)

        for device in self._ordered_candidates():
            if device == current_family:
                continue
            if self._is_quarantined(device, now):
                continue
            return device

        # Last resort: CPU, even if quarantined, provided we are not already on it.
        if "CPU" in self._usable and current_family != "CPU":
            return "CPU"
        return None

    def preferred_upgrade(
        self, current: str, now: Optional[float] = None
    ) -> Optional[str]:
        """The best healthy family that *outranks* ``current``, or ``None``.

        Unlike :meth:`next_device` (which flees a bad device in any direction),
        this only ever points upwards in the priority order -- it is how work
        drifts back onto a recovered accelerator once its quarantine expires,
        without waiting for an overload to force the issue.
        """

        now = time.monotonic() if now is None else now
        current_family = _family(current)
        for device in self._ordered_candidates():
            if device == current_family:
                return None  # already on the best available
            if self._is_quarantined(device, now):
                continue
            return device
        return None


class FallbackController:
    """Turns failure/overload signals into concrete device switches."""

    def __init__(
        self,
        config: FallbackConfig,
        device_manager: DeviceManager,
        detector: FaceDetector,
        policy: FallbackPolicy,
    ) -> None:
        self._config = config
        self._device_manager = device_manager
        self._detector = detector
        self._policy = policy
        self._consecutive_failures = 0
        # Monotonic timestamp before which no new switch attempt is made.
        self._next_switch_allowed_at = 0.0
        # "No fallback available" is logged loudly once, then suppressed until
        # a switch target exists again (prevents an error line every backoff
        # window for hours under a persistently-breached threshold).
        self._no_target_logged = False
        # Next time the periodic preferred-device upgrade check may run.
        self._next_upgrade_check_at = 0.0
        self._lock = threading.Lock()

        # Emitted after a successful fallback switch: (from_device, to_device).
        self.fallback_occurred = Event("fallback_occurred")

    # ------------------------------------------------------------------
    # Signal handlers (called from the inference / monitor threads)
    # ------------------------------------------------------------------
    def handle_inference_error(self, error: Exception) -> None:
        """Record an inference failure and switch device once the threshold is hit.

        Errors that indicate the device itself is gone (driver reset, context
        lost, ...) bypass the threshold entirely: waiting for more failures on a
        dead device only adds delay -- each doomed call can block for seconds in
        the driver -- so we flee to the next device on the first such error.
        """

        with self._lock:
            self._consecutive_failures += 1
            if _looks_fatal(error):
                _log.warning(
                    "Fatal device error, switching away immediately: %s", error
                )
                self._consecutive_failures = 0
                self._switch_away(reason="fatal device error", escalate=True)
                return
            # Per-frame failures are DEBUG-level to avoid flooding the log at high
            # frame rates; the switch decision below is logged at INFO/WARNING.
            _log.debug(
                "Inference failure %d/%d: %s",
                self._consecutive_failures,
                self._config.max_consecutive_failures,
                error,
            )
            if self._consecutive_failures < self._config.max_consecutive_failures:
                return
            # Consume the streak so we make at most one switch decision per
            # ``max_consecutive_failures`` frames (backoff throttles it further).
            self._consecutive_failures = 0
            self._switch_away(reason="repeated inference errors", escalate=True)

    def handle_overload(self, detail: str) -> None:
        """React to a resource-overload signal from the monitor.

        Only acts when the saturated resource is the one running inference (or
        the latency signal says the active device itself is degraded) -- see
        :func:`overload_concerns_device`. A CPU breach while inference runs on
        the GPU is somebody else's workload; moving onto the busy CPU would
        make things worse.
        """

        if not self._config.switch_on_overload:
            return
        with self._lock:
            current = self._device_manager.effective_device()
            if not overload_concerns_device(detail, _family(current)):
                _log.debug(
                    "Overload (%s) does not concern active device '%s'; staying",
                    detail,
                    current,
                )
                return
            _log.warning("Overload detected (%s); considering fallback", detail)
            self._switch_away(reason=f"overload: {detail}")

    def maybe_upgrade(self, *_args) -> None:
        """Move inference back onto a recovered higher-priority device.

        Driven by the monitor's per-second sample event and self-rate-limited
        to ``preferred_retry_sec``. This is deliberately independent of
        overload signals: a GPU that crashed and later recovered should get
        the work back even when the CPU is coping fine.
        """

        if self._config.preferred_retry_sec <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_upgrade_check_at:
                return
            self._next_upgrade_check_at = now + self._config.preferred_retry_sec
            current = self._device_manager.effective_device()
            target = self._policy.preferred_upgrade(current, now=now)
            if target is None:
                return
            _log.info(
                "Preferred device '%s' is available again; switching back from %s",
                target,
                current,
            )
            if not self._apply_switch(target):
                # Refused to compile: keep it out of rotation a while so the
                # periodic check does not hammer a still-broken device.
                self._policy.quarantine(target, now=now, escalate=True)

    def notify_success(self) -> None:
        """Reset the failure counter after a healthy inference.

        Also forgives the current device's escalation strikes: it erred below
        the switch threshold and then recovered on its own, so it should not be
        punished with a longer quarantine the next time it stumbles.
        """

        if self._consecutive_failures:
            with self._lock:
                self._consecutive_failures = 0
                self._policy.clear_strikes(self._device_manager.effective_device())

    # ------------------------------------------------------------------
    # Explicit switching (used by the app's public API / future GUI)
    # ------------------------------------------------------------------
    def switch_to(self, device: str) -> bool:
        """Force a switch onto ``device``. Returns ``True`` on success."""

        with self._lock:
            return self._apply_switch(device)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _switch_away(self, reason: str, escalate: bool = False) -> None:
        """Quarantine the current device and move to the policy's next choice.

        A time-based backoff gates how often a switch is even attempted, so a
        device that fails continuously cannot cause the controller to thrash or
        spam the log. ``escalate`` marks an *error* quarantine, whose cooldown
        grows on repeat offences (overload quarantines stay flat). Assumes the
        caller holds the lock.
        """

        now = time.monotonic()
        if now < self._next_switch_allowed_at:
            return  # still within the backoff window from a recent attempt
        # Whatever happens below counts as an attempt; schedule the next window.
        self._next_switch_allowed_at = now + self._config.switch_backoff_sec

        current = self._device_manager.effective_device()
        target = self._policy.next_device(current, now=now)
        if target is None:
            # Nowhere to go: stay put and do NOT quarantine the current device
            # -- it is the only one we have, and marking it unusable would only
            # block legitimate switches back onto it later.
            if not self._no_target_logged:
                self._no_target_logged = True
                _log.error(
                    "No fallback device available (%s); will keep checking "
                    "quietly and switch when one frees up",
                    reason,
                )
            else:
                _log.debug("Still no fallback device (%s)", reason)
            return
        self._no_target_logged = False

        self._policy.quarantine(current, now=now, escalate=escalate)
        if escalate:
            # The device is being fled because it *errored*: any model compiled
            # on its (possibly reset) context is suspect, so a later switch back
            # must recompile rather than reuse a cached standby.
            self._device_manager.invalidate_standby(_family(current))

        previous = current
        if self._apply_switch(target):
            _log.info(
                "Application-level fallback: %s -> %s (%s)", previous, target, reason
            )
            self.fallback_occurred.emit(previous, target)

    def _apply_switch(self, target: str) -> bool:
        """Recompile onto ``target`` and rebind the detector. Assumes lock held.

        Shared by both fallback (:meth:`_switch_away`) and explicit user switches
        (:meth:`switch_to`); the ``fallback_occurred`` event is fired only by the
        former, since a deliberate switch is not a fallback.
        """

        try:
            compiled = self._device_manager.switch_device(target)
            self._detector.set_compiled_model(compiled)
        except DeviceError as exc:
            # Native plugin errors span many lines; keep the log to the point.
            _log.error("Switch to '%s' failed: %s", target, str(exc).splitlines()[0])
            _log.debug("Full switch failure detail for '%s': %s", target, exc)
            return False

        self._consecutive_failures = 0
        return True
