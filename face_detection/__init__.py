"""Adaptive face detection on the OpenVINO AUTO plugin.

Public API for embedding the backend (e.g. from main.py or a UI).
"""

import os as _os

# Quiet OpenCV's native (C++) videoio logging before cv2 is imported anywhere in
# the package. A failing/reset camera backend (e.g. MSMF on Windows) otherwise
# prints a warning to stderr on *every* failed grab, which we cannot throttle
# from Python. Overridable via the environment if deeper diagnostics are needed.
_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from .app import Application, ApplicationError
from .config import AppConfig, load_config

# Logged at start-up so any pasted log identifies exactly which build produced
# it -- the project is copied between machines, and "which code is this?" has
# to be answerable from the log alone. Bump when behaviour changes.
__version__ = "2026-07-02.10"

__all__ = ["Application", "ApplicationError", "AppConfig", "load_config", "__version__"]
