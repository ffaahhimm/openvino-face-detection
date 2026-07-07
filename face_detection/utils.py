"""Generic, dependency-free helpers shared across modules.

Only truly cross-cutting utilities belong here -- anything tied to a specific
domain (cameras, inference, devices) lives in its own module. Keeping this file
small and side-effect-free avoids the "junk drawer" anti-pattern.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional


class FPSCounter:
    """Rolling frames-per-second estimator.

    Uses a fixed-size window of frame timestamps so the reported rate reacts
    quickly to changes (e.g. after a device switch) without being noisy.
    """

    def __init__(self, window: int = 30, stall_after_sec: float = 2.0) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window)
        self._stall_after_sec = stall_after_sec

    def tick(self) -> None:
        """Record that one frame has just been processed."""

        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        """Current FPS averaged over the window (0.0 until enough samples).

        Returns 0.0 if no frame has arrived recently, so a stalled pipeline
        (e.g. the camera stopped yielding) honestly reads as 0 rather than
        freezing at the last computed rate.
        """

        if len(self._timestamps) < 2:
            return 0.0
        if time.monotonic() - self._timestamps[-1] > self._stall_after_sec:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


class MovingAverage:
    """Simple windowed moving average, used for smoothing inference times."""

    def __init__(self, window: int = 30) -> None:
        self._values: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def last(self) -> Optional[float]:
        return self._values[-1] if self._values else None

    def reset(self) -> None:
        self._values.clear()


class Stopwatch:
    """Context manager that measures wall-clock elapsed time in milliseconds.

    Example
    -------
    >>> with Stopwatch() as sw:
    ...     do_work()
    >>> print(sw.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
