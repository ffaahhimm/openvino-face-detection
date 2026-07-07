"""Console reporting of live pipeline statistics.

Fulfils the requirement to "display FPS, inference time, and the currently
active device in the console" while respecting "use structured logging instead
of ``print()``": the reporter is a lightweight thread that periodically formats
the latest :class:`~pipeline.PipelineStats` and :class:`~monitor.ResourceSnapshot`
into a single structured log line.

Keeping display concerns in their own module means the pipeline never has to
know or care how (or whether) stats are shown -- a GUI could ignore this
reporter entirely and subscribe to the raw events instead.
"""

from __future__ import annotations

import threading

from .logger import get_logger
from .monitor import ResourceMonitor
from .pipeline import PipelineStats

_log = get_logger("stats")


class ConsoleReporter(threading.Thread):
    """Periodically logs a one-line summary of pipeline + resource stats."""

    def __init__(
        self,
        stats: PipelineStats,
        monitor: ResourceMonitor,
        interval_sec: float,
    ) -> None:
        super().__init__(name="ConsoleReporter", daemon=True)
        self._stats = stats
        self._monitor = monitor
        self._interval_sec = interval_sec
        self._stop_event = threading.Event()

    def run(self) -> None:
        _log.info("Console reporter started (every %.1fs)", self._interval_sec)
        while not self._stop_event.wait(self._interval_sec):
            self._report()
        _log.info("Console reporter stopped")

    def _report(self) -> None:
        stats = self._stats.snapshot()
        resources = self._monitor.latest()

        gpu = "n/a"
        cpu = "n/a"
        mem = "n/a"
        if resources is not None:
            cpu = f"{resources.cpu_percent:4.1f}%"
            mem = f"{resources.memory_percent:4.1f}%"
            gpu = "n/a" if resources.gpu_percent is None else f"{resources.gpu_percent:4.1f}%"

        _log.info(
            "device=%-12s | fps=%5.1f | infer=%6.1fms (avg %6.1fms) | "
            "faces=%d | frames=%d | errors=%d | cpu=%s mem=%s gpu=%s",
            stats.device,
            stats.fps,
            stats.last_inference_ms,
            stats.avg_inference_ms,
            stats.faces_detected,
            stats.frames_processed,
            stats.errors,
            cpu,
            mem,
            gpu,
        )

    def stop(self) -> None:
        self._stop_event.set()
