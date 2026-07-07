"""A tiny, thread-safe observer/event bus.

This module is the seam that keeps the backend GUI-ready. Core components emit
events (new frame, new detections, device changed, resource sample); a future
Tkinter/Qt front-end simply subscribes to the ones it cares about -- no core
logic has to change to add a UI.

The implementation is deliberately minimal: no external dependency, exceptions
raised by one subscriber never break another, and subscription is safe from any
thread.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List

from .logger import get_logger

_log = get_logger(__name__)

Subscriber = Callable[..., None]


class Event:
    """A multicast callback slot.

    Subscribers are invoked synchronously in registration order on whichever
    thread calls :meth:`emit`. Handlers are expected to be cheap; anything heavy
    (e.g. rendering) should hand work off to its own queue/thread.
    """

    def __init__(self, name: str = "event") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._subscribers: List[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> Subscriber:
        """Register ``callback``; returns it so it can be used as a decorator."""

        with self._lock:
            self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: Subscriber) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        """Invoke every subscriber, isolating failures so one bad handler
        cannot take down the pipeline thread that fired the event."""

        # Copy under the lock so subscribers may (un)subscribe during emission.
        with self._lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(*args, **kwargs)
            except Exception:  # noqa: BLE001 - isolation is the whole point
                _log.exception("Subscriber of event '%s' raised; ignoring", self._name)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
