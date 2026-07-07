"""Frame annotation helpers (OpenCV drawing).

Shared by the optional ``preview.py`` window and the ``gui.py`` Tkinter app so
neither duplicates the box-drawing code. This module only draws onto NumPy
frames; it knows nothing about any UI toolkit.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from .inference import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)      # green (BGR)
_HUD_COLOR: Tuple[int, int, int] = (0, 255, 255)    # yellow (BGR)


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    color: Tuple[int, int, int] = _BOX_COLOR,
) -> np.ndarray:
    """Draw a box + confidence label for each detection (in place). Returns frame."""

    for det in detections:
        cv2.rectangle(frame, (det.x_min, det.y_min), (det.x_max, det.y_max), color, 2)
        cv2.putText(
            frame,
            f"{det.confidence:.2f}",
            (det.x_min, max(12, det.y_min - 6)),
            _FONT,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray, text: str, color: Tuple[int, int, int] = _HUD_COLOR
) -> np.ndarray:
    """Draw a single-line heads-up overlay at the top-left (in place).

    The text is stroked with a black outline so it stays readable over both dark
    and bright parts of the frame -- no background bar needed.
    """

    org = (12, 30)
    cv2.putText(frame, text, org, _FONT, 0.7, (0, 0, 0), 5, cv2.LINE_AA)  # outline
    cv2.putText(frame, text, org, _FONT, 0.7, color, 2, cv2.LINE_AA)      # fill
    return frame
