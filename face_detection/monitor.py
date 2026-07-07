"""Resource monitoring.

A background thread samples system load at a fixed interval and publishes each
sample through an :class:`~events.Event`. It also implements simple hysteresis:
only after several *consecutive* over-threshold samples does it raise an
``on_overload`` event, which the :class:`~fallback.FallbackController` uses to
migrate work off a saturated device.

GPU utilisation is genuinely hardware/vendor specific and ``psutil`` does not
expose it. We therefore treat GPU monitoring as *best-effort*: if the optional
``pynvml`` (NVIDIA) library is importable we use it, otherwise GPU load is
reported as ``None`` and everything else keeps working.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import psutil

from .config import MonitorConfig
from .events import Event
from .logger import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class ResourceSnapshot:
    """A single point-in-time reading of system resources."""

    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float]  # None when GPU telemetry is unavailable


class _PdhGpuProbe:
    """Windows GPU utilisation via the 'GPU Engine' performance counters.

    These are the counters Task Manager's GPU graph reads, so Intel / AMD /
    NVIDIA adapters are all visible with no extra packages -- which matters
    here because OpenVINO executes on the *Intel* GPU, invisible to NVML.

    Attribution: OpenVINO's GPU plugin runs on Intel silicon only, so when
    Intel adapters exist their engines alone are measured (counter instance
    names embed each adapter's LUID; the DirectX registry key maps LUIDs to
    vendors). A discrete NVIDIA card busy with something else can therefore
    never masquerade as inference-GPU load. Utilisation is the busiest
    matching engine's percentage. Rate counters need two collections; the
    first sample returns None/0.
    """

    _PDH_FMT_DOUBLE = 0x00000200
    _PDH_MORE_DATA = 0x800007D2
    _INTEL_VENDOR_ID = 0x8086

    class _CounterValue(ctypes.Structure):
        _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]

    class _CounterItem(ctypes.Structure):
        pass  # _fields_ set below (needs _CounterValue defined first)

    def __init__(self) -> None:
        self._pdh = None
        self._query = None
        self._counter = None
        self._intel_luids: list = []
        try:
            self._CounterItem._fields_ = [
                ("szName", ctypes.c_wchar_p),
                ("FmtValue", self._CounterValue),
            ]
        except Exception:  # noqa: BLE001 - already set on a previous init
            pass
        try:
            pdh = ctypes.windll.pdh
            query = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return
            counter = ctypes.c_void_p()
            path = r"\GPU Engine(*)\Utilization Percentage"
            if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) != 0:
                pdh.PdhCloseQuery(query)
                return
            pdh.PdhCollectQueryData(query)  # prime the rate counters
            self._pdh, self._query, self._counter = pdh, query, counter
            self._intel_luids = self._intel_adapter_luids()
        except Exception:  # noqa: BLE001 - PDH unavailable: stay unsupported
            self._pdh = None

    def _intel_adapter_luids(self) -> list:
        """LUID substrings (as used in counter instance names) of Intel GPUs.

        Read from ``HKLM\\SOFTWARE\\Microsoft\\DirectX``, where Windows records
        every adapter's LUID, vendor and description. Stale entries from
        previous boots hold LUIDs that match no live counter instance, so they
        are harmless. Empty when nothing Intel is found -- the probe then
        measures all adapters rather than nothing.
        """

        luids = []
        try:
            import winreg

            root = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\DirectX"
            )
            index = 0
            while True:
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(root, name) as key:
                        vendor, _ = winreg.QueryValueEx(key, "VendorId")
                        if vendor != self._INTEL_VENDOR_ID:
                            continue
                        luid, _ = winreg.QueryValueEx(key, "AdapterLuid")
                        luids.append(
                            "luid_0x%08x_0x%08x"
                            % ((luid >> 32) & 0xFFFFFFFF, luid & 0xFFFFFFFF)
                        )
                except OSError:
                    continue
            winreg.CloseKey(root)
        except Exception:  # noqa: BLE001 - registry shape changed: fall back to all
            return []
        return luids

    @property
    def supported(self) -> bool:
        return self._pdh is not None

    def utilisation(self) -> Optional[float]:
        if not self.supported:
            return None
        try:
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                return None
            size = ctypes.c_ulong(0)
            count = ctypes.c_ulong(0)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._counter, self._PDH_FMT_DOUBLE,
                ctypes.byref(size), ctypes.byref(count), None,
            ) & 0xFFFFFFFF  # PDH_STATUS is unsigned; ctypes returns signed int
            if status != self._PDH_MORE_DATA:
                return None
            buffer = (ctypes.c_byte * size.value)()
            if self._pdh.PdhGetFormattedCounterArrayW(
                self._counter, self._PDH_FMT_DOUBLE,
                ctypes.byref(size), ctypes.byref(count), buffer,
            ) != 0:
                return None
            items = ctypes.cast(
                buffer, ctypes.POINTER(self._CounterItem * count.value)
            ).contents
            # The busiest single engine is what saturation feels like; values
            # are per-engine so summing could exceed 100% meaninglessly. When
            # Intel adapters are known, only their engines count -- that is
            # the silicon OpenVINO's GPU plugin actually runs on.
            busiest = 0.0
            for item in items:
                if item.FmtValue.CStatus != 0:
                    continue
                if self._intel_luids:
                    name = (item.szName or "").lower()
                    if not any(luid in name for luid in self._intel_luids):
                        continue
                busiest = max(busiest, item.FmtValue.doubleValue)
            return min(100.0, busiest)
        except Exception:  # noqa: BLE001
            return None

    def shutdown(self) -> None:
        if self._pdh is not None:
            try:
                self._pdh.PdhCloseQuery(self._query)
            except Exception:  # noqa: BLE001
                pass


class _GpuProbe:
    """Best-effort GPU utilisation, isolated so the monitor stays agnostic.

    On Windows the vendor-neutral performance counters are preferred (they see
    the Intel GPU OpenVINO actually runs on); NVML is the fallback and the
    only option elsewhere. All failures degrade gracefully to "unsupported".
    """

    def __init__(self) -> None:
        self._handle = None
        self._pynvml = None
        self._pdh: Optional[_PdhGpuProbe] = None

        if sys.platform == "win32":
            probe = _PdhGpuProbe()
            if probe.supported:
                self._pdh = probe
                if probe._intel_luids:
                    _log.info(
                        "GPU monitoring enabled via Windows performance counters "
                        "(scoped to %d Intel adapter(s) -- the silicon OpenVINO uses)",
                        len(probe._intel_luids),
                    )
                else:
                    _log.info(
                        "GPU monitoring enabled via Windows performance counters "
                        "(no Intel adapter found in registry; measuring all adapters)"
                    )
                return

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _log.info("GPU monitoring enabled via NVML")
        except Exception:  # noqa: BLE001 - optional dependency / no NVIDIA GPU
            _log.info(
                "GPU monitoring unavailable (no performance counters, and "
                "pynvml/NVIDIA GPU not present)"
            )

    @property
    def supported(self) -> bool:
        return self._pdh is not None or self._handle is not None

    def utilisation(self) -> Optional[float]:
        if self._pdh is not None:
            return self._pdh.utilisation()
        if self._handle is None:
            return None
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception:  # noqa: BLE001
            return None

    def shutdown(self) -> None:
        if self._pdh is not None:
            self._pdh.shutdown()
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


class ResourceMonitor(threading.Thread):
    """Periodically samples CPU / memory / GPU and detects sustained overload."""

    def __init__(
        self,
        config: MonitorConfig,
        latency_provider: Optional["Callable[[], float]"] = None,
    ) -> None:
        super().__init__(name="ResourceMonitor", daemon=True)
        self._config = config
        # Returns the active device's average inference time (ms); used to detect
        # a device that is overloaded/degraded even when system CPU/RAM look fine.
        self._latency_provider = latency_provider
        self._stop_event = threading.Event()
        self._gpu = _GpuProbe()
        self._latest: Optional[ResourceSnapshot] = None
        self._latest_lock = threading.Lock()
        self._overload_streak = 0

        # Subscribers: the reporter (display) and fallback controller (overload).
        self.on_sample = Event("resource_sample")
        self.on_overload = Event("resource_overload")

    def run(self) -> None:
        _log.info("Resource monitor started (interval=%.1fs)", self._config.interval_sec)
        # Prime psutil so the first cpu_percent reading is meaningful, not 0.0.
        psutil.cpu_percent(interval=None)

        while not self._stop_event.wait(self._config.interval_sec):
            snapshot = self._sample()
            with self._latest_lock:
                self._latest = snapshot
            self.on_sample.emit(snapshot)
            self._check_overload(snapshot)

        self._gpu.shutdown()
        _log.info("Resource monitor stopped")

    def _sample(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
            gpu_percent=self._gpu.utilisation(),
        )

    def _check_overload(self, snapshot: ResourceSnapshot) -> None:
        """Raise ``on_overload`` after ``overload_samples`` consecutive breaches."""

        breaches = []
        if snapshot.cpu_percent >= self._config.cpu_threshold:
            breaches.append(f"CPU {snapshot.cpu_percent:.0f}%")
        if snapshot.memory_percent >= self._config.memory_threshold:
            breaches.append(f"MEM {snapshot.memory_percent:.0f}%")
        if (
            snapshot.gpu_percent is not None
            and snapshot.gpu_percent >= self._config.gpu_threshold
        ):
            breaches.append(f"GPU {snapshot.gpu_percent:.0f}%")

        # Device-level overload: the active device's inference is too slow. This
        # is the signal that most directly means "this device is struggling".
        threshold = self._config.inference_latency_threshold_ms
        if self._latency_provider is not None and threshold > 0:
            latency = self._latency_provider()
            if latency >= threshold:
                breaches.append(f"LATENCY {latency:.0f}ms")

        if not breaches:
            self._overload_streak = 0
            return

        self._overload_streak += 1
        if self._overload_streak >= self._config.overload_samples:
            self._overload_streak = 0  # reset so we don't spam every sample
            self.on_overload.emit(", ".join(breaches))

    def latest(self) -> Optional[ResourceSnapshot]:
        """Most recent snapshot, or ``None`` before the first sample."""

        with self._latest_lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()
