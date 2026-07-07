"""Optional live viewer — the camera feed only, with a device + FPS overlay.

Deliberately minimal: an OpenCV window sized exactly to the frame (no side
panels, no menu bars, no wasted space). The active device and current FPS are
drawn directly onto the video. Like every front-end in this project it attaches
to the backend purely through the public ``Application.on_result`` event, so no
core module changes.

Threading model
---------------
OpenCV's HighGUI (``imshow`` / ``waitKey``) must run on the main thread, so we
run :meth:`Application.run` on a background thread and keep the display loop on
the main thread. The inference thread only hands us the latest frame through a
thread-safe slot.

Run:
    python gui.py        # press 'q' or Esc, or close the window, to quit
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .app import Application
from .config import AppConfig, load_config
from .logger import get_logger, setup_logging
from .visualization import draw_detections, draw_hud

_WINDOW = "Face Detection (OpenVINO AUTO)"
_QUIT_KEYS = (ord("q"), 27)  # 'q' or Esc


def _placeholder(width: int, height: int, text: str = "Starting camera + model...") -> np.ndarray:
    """A frame shown immediately at launch so the window appears without delay."""

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
    return frame


class _Slot:
    """Thread-safe single slot holding the most recent (frame, detections, stats)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


def _annotate(frame, detections, stats):
    """Draw face boxes plus a compact 'device | FPS' overlay onto a frame copy."""

    annotated = draw_detections(frame.copy(), detections)
    return draw_hud(annotated, f"{stats.device}  |  {stats.fps:.1f} FPS")


def run(config: AppConfig) -> None:
    """Launch the OpenCV overlay viewer for the given configuration.

    Reusable so ``main.py`` can start the GUI without duplicating the setup.
    """

    log = get_logger("gui")

    app = Application(config)
    slot = _Slot()
    app.on_result.subscribe(lambda frame, detections, stats: slot.set((frame, detections, stats)))

    startup_error: dict = {}

    def _run() -> None:
        try:
            app.run()
        except Exception as exc:  # noqa: BLE001 - reported to the display loop
            startup_error["exc"] = exc

    runner = threading.Thread(target=_run, name="AppRunner", daemon=True)
    runner.start()

    # AUTOSIZE => the window is exactly the frame size: feed only, no extra space.
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    # Show a placeholder right away so the window appears instantly instead of
    # looking hung while the camera and model start up.
    starting = _placeholder(config.camera.width, config.camera.height)
    log.info("Live view open; press 'q'/Esc or close the window to stop.")
    try:
        while runner.is_alive():
            data = slot.get()
            if data is None:
                # Still starting up: show the placeholder, keep the UI responsive.
                cv2.imshow(_WINDOW, starting)
                if (cv2.waitKey(30) & 0xFF) in _QUIT_KEYS:
                    break
                continue

            frame, detections, stats = data
            cv2.imshow(_WINDOW, _annotate(frame, detections, stats))

            if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                break
            # Stop if the user closed the window with the title-bar button.
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # shutdown() performs a bounded, Ctrl-C-safe teardown; run() then returns.
        app.shutdown()
        runner.join(timeout=3.0)
        cv2.destroyAllWindows()

    if startup_error:
        log.error("Startup failed: %s", startup_error["exc"])


def main() -> None:
    config = load_config()
    setup_logging(config.logging)
    run(config)


if __name__ == "__main__":
    main()
