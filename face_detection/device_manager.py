"""OpenVINO device management and the AUTO-plugin integration.

This module owns the single :class:`openvino.Core` instance and is the *only*
place that talks to the OpenVINO runtime about devices. It is responsible for:

* **Device discovery** -- enumerating usable CPU / GPU / NPU devices.
* **AUTO-plugin compilation** -- building the ``AUTO:GPU,NPU,CPU`` candidate
  string and compiling the model through the AUTO plugin.
* **Runtime device switching** -- recompiling the model onto a specific device
  on demand, without restarting the process.

--------------------------------------------------------------------------
How the OpenVINO AUTO plugin fits in
--------------------------------------------------------------------------
The AUTO plugin is a *virtual* device. Instead of compiling for, say, "GPU",
we compile for ``AUTO:GPU,NPU,CPU``. AUTO then:

1. Immediately starts inference on CPU (fast to load) while the accelerator
   compiles in the background, then transparently migrates to the first
   available high-priority device (GPU, then NPU).
2. Provides a *first line* of hardware fallback: if a listed device is missing
   or fails to load, AUTO picks the next candidate on its own.

On top of that, this application adds a *second line* of fallback (see
``fallback.py``): if the active device starts throwing errors or the machine is
overloaded at runtime, we explicitly recompile onto a different device. The two
mechanisms are complementary -- AUTO handles load-time selection, our
controller handles run-time recovery.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import openvino as ov

from .config import DeviceConfig
from .events import Event
from .logger import get_logger

_log = get_logger(__name__)

# Devices this application knows how to reason about. Anything else reported by
# OpenVINO (e.g. "GNA") is ignored for scheduling purposes.
_KNOWN_FAMILIES = ("CPU", "GPU", "NPU")

# The AUTO plugin's virtual device name.
AUTO_DEVICE = "AUTO"


class DeviceError(RuntimeError):
    """Raised when a model cannot be compiled onto the requested device."""


def _family(device_name: str) -> str:
    """Return the base family for a device id (``"GPU.1"`` -> ``"GPU"``)."""

    return device_name.split(".", 1)[0].upper()


class DeviceManager:
    """Owns the OpenVINO ``Core`` and mediates all device operations.

    Thread-safety: compilation and device switching are guarded by a lock, so
    the monitor/fallback threads can request a switch while the inference thread
    is running.
    """

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._core = ov.Core()
        # On-disk kernel cache: makes (re)compilation much faster, which matters
        # most on GPU and when a supervised worker restarts.
        if config.cache_dir:
            try:
                self._core.set_property({"CACHE_DIR": config.cache_dir})
            except Exception:  # noqa: BLE001 - caching is a best-effort optimisation
                _log.debug("Could not set CACHE_DIR=%s", config.cache_dir)
        self._model: Optional[ov.Model] = None
        self._compiled: Optional[ov.CompiledModel] = None
        self._current_device: str = AUTO_DEVICE
        # Hot standbys: device -> model already compiled for it, so a fallback
        # switch is an instant rebind instead of a (re)compilation.
        self._standby: dict[str, ov.CompiledModel] = {}
        self._lock = threading.RLock()

        # Emitted whenever the *effective* execution device changes. GUIs and
        # the console reporter subscribe to this.
        self.device_changed = Event("device_changed")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def available_devices(self) -> List[str]:
        """Return the raw device ids OpenVINO reports (e.g. ``["CPU", "GPU"]``)."""

        return list(self._core.available_devices)

    def usable_devices(self) -> List[str]:
        """Return known-family devices that policy permits, deduplicated by family.

        Honours ``allow_gpu`` / ``allow_npu`` from configuration and always keeps
        CPU as the guaranteed final fallback.
        """

        allowed: List[str] = []
        seen_families: set[str] = set()
        for device in self.available_devices():
            family = _family(device)
            if family not in _KNOWN_FAMILIES:
                continue
            if family == "GPU" and not self._config.allow_gpu:
                continue
            if family == "NPU" and not self._config.allow_npu:
                continue
            if family in seen_families:
                continue
            seen_families.add(family)
            allowed.append(family)
        return allowed

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def load_model(self, xml_path: str, bin_path: str = "") -> None:
        """Read the IR model from disk (once). Raises :class:`DeviceError`."""

        try:
            if bin_path:
                self._model = self._core.read_model(model=xml_path, weights=bin_path)
            else:
                self._model = self._core.read_model(model=xml_path)
        except Exception as exc:  # noqa: BLE001 - normalise to our error type
            raise DeviceError(f"Failed to read model '{xml_path}': {exc}") from exc
        _log.info("Model loaded from %s", xml_path)

    def build_auto_string(self) -> str:
        """Construct the AUTO candidate string from configured priority.

        Only devices actually present are included; CPU is appended as a safety
        net if it is not already listed. Example result: ``"AUTO:GPU,CPU"``.
        """

        usable = self.usable_devices()
        ordered = [d for d in self._config.auto_priority if d in usable]
        if "CPU" not in ordered and "CPU" in usable:
            ordered.append("CPU")
        if not ordered:
            # Nothing matched -- let AUTO decide entirely on its own.
            return AUTO_DEVICE
        return f"{AUTO_DEVICE}:{','.join(ordered)}"

    def compile_auto(self) -> ov.CompiledModel:
        """Compile the model through the AUTO plugin (the default path)."""

        return self._compile(self.build_auto_string())

    def compile_for(self, device_name: str) -> ov.CompiledModel:
        """Compile the model for an explicit device (used by runtime switching)."""

        return self._compile(device_name)

    def _compile(self, device_name: str) -> ov.CompiledModel:
        if self._model is None:
            raise DeviceError("compile requested before load_model()")

        config = {"PERFORMANCE_HINT": self._config.performance_hint}
        with self._lock:
            try:
                compiled = self._core.compile_model(
                    model=self._model, device_name=device_name, config=config
                )
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(
                    f"Failed to compile model on '{device_name}': {exc}"
                ) from exc

            self._compiled = compiled
            self._current_device = device_name
            # Concrete-device compiles double as standbys, so bouncing between
            # devices (e.g. overload-driven CPU <-> GPU) recompiles at most once
            # per device. AUTO strings are excluded: AUTO re-selects each time.
            if not device_name.upper().startswith(AUTO_DEVICE):
                self._standby[device_name.upper()] = compiled
            effective = self._read_execution_devices(compiled, device_name)
            _log.info(
                "Model compiled on '%s' (executing on: %s)", device_name, effective
            )

        self.device_changed.emit(device_name, effective)
        return compiled

    def switch_device(self, device_name: str) -> ov.CompiledModel:
        """Move inference onto ``device_name`` at runtime and return the new model.

        This is the mechanism behind "runtime device switching without a
        restart": callers swap the returned :class:`ov.CompiledModel` into the
        :class:`~inference.FaceDetector`. If a hot standby exists for the device
        (see :meth:`prepare_standby`) it is used directly, making the switch
        effectively instantaneous -- crucial when fleeing a failing device.
        """

        with self._lock:
            standby = self._standby.get(device_name.upper())
            if standby is not None:
                _log.info("Switching device -> %s (using pre-compiled standby)", device_name)
                self._compiled = standby
                self._current_device = device_name
                effective = self._read_execution_devices(standby, device_name)
        if standby is not None:
            self.device_changed.emit(device_name, effective)
            return standby

        _log.info("Switching device -> %s", device_name)
        return self.compile_for(device_name)

    def prepare_standby(self, device_name: str) -> bool:
        """Compile the model for ``device_name`` and keep it warm for fallback.

        Unlike :meth:`compile_for`, this does *not* make the device current and
        fires no events -- it only ensures that a later :meth:`switch_device`
        onto it needs no compilation. Intended to be called on a background
        thread at start-up (typically for CPU, the guaranteed last resort).
        Returns ``True`` if a standby is ready.
        """

        device_name = device_name.upper()
        with self._lock:
            if device_name in self._standby:
                return True
        if self._model is None:
            _log.warning("Standby for '%s' requested before load_model()", device_name)
            return False
        config = {"PERFORMANCE_HINT": self._config.performance_hint}
        try:
            compiled = self._core.compile_model(
                model=self._model, device_name=device_name, config=config
            )
        except Exception as exc:  # noqa: BLE001 - standby is an optimisation, not a requirement
            _log.warning("Could not prepare standby on '%s': %s", device_name, exc)
            return False
        with self._lock:
            self._standby[device_name] = compiled
        _log.info("Standby model ready on '%s' (instant fallback target)", device_name)
        return True

    def invalidate_standby(self, device_name: str) -> None:
        """Drop the cached standby for ``device_name``.

        Called when a device is fled because of *errors*: a model compiled on a
        crashed/reset device context must not be handed back out on a later
        switch -- it gets recompiled from scratch instead.
        """

        with self._lock:
            if self._standby.pop(device_name.upper(), None) is not None:
                _log.info("Standby on '%s' invalidated after device errors", device_name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def _read_execution_devices(
        self, compiled: ov.CompiledModel, requested: str
    ) -> str:
        """Return the concrete device(s) AUTO actually scheduled work onto.

        The AUTO plugin sometimes formats the value with parentheses
        (e.g. ``"(CPU)"``); we normalise that away so the string is both
        display-friendly and safe for the fallback policy's family matching.
        """

        try:
            devices = compiled.get_property("EXECUTION_DEVICES")
            if isinstance(devices, (list, tuple)):
                tokens = [self._clean_device_token(str(d)) for d in devices]
            else:
                tokens = [self._clean_device_token(str(devices))]
            tokens = [t for t in tokens if t]
            return ", ".join(tokens) if tokens else requested
        except Exception:  # noqa: BLE001 - property is best-effort
            return requested

    @staticmethod
    def _clean_device_token(token: str) -> str:
        """Strip AUTO's decorative parentheses/whitespace from a device id."""

        return token.replace("(", "").replace(")", "").strip()

    @property
    def current_device(self) -> str:
        """The device string most recently compiled for (may be an AUTO string)."""

        with self._lock:
            return self._current_device

    def effective_device(self) -> str:
        """The concrete hardware currently executing inference, if known."""

        with self._lock:
            if self._compiled is None:
                return "N/A"
            return self._read_execution_devices(self._compiled, self._current_device)
