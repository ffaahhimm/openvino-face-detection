"""Entry point.

By default this launches the live GUI viewer (camera feed with a device + FPS
overlay) with inference in an isolated worker process, so a device hang or
crash never takes the application down. Flags:

* ``--headless`` -- run the backend with structured console output, no window.
* ``--process``  -- force process-isolated inference (already the default).
* ``--thread``   -- run inference in-process instead: lowest latency and
  runtime device switching, but no immunity to native device crashes.

All real work lives in the dedicated modules; this file just wires the chosen
front-end to the configuration and handles graceful shutdown.
"""

import argparse
import multiprocessing
from dataclasses import replace

from face_detection.app import Application, ApplicationError
from face_detection.config import load_config
from face_detection.logger import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO AUTO face detection")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the GUI (structured console output only)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--process",
        action="store_true",
        help="run inference in an isolated worker process (default; survives device hang/crash)",
    )
    mode_group.add_argument(
        "--thread",
        action="store_true",
        help="run inference in-process (lowest latency, no crash isolation)",
    )
    args = parser.parse_args()

    config = load_config()
    if args.process:
        config = replace(config, inference=replace(config.inference, mode="process"))
    elif args.thread:
        config = replace(config, inference=replace(config.inference, mode="thread"))
    setup_logging(config.logging)
    log = get_logger("main")

    try:
        if args.headless:
            _run_headless(config)
        else:
            # Imported lazily so headless mode never needs the GUI stack.
            from face_detection.gui import run as run_gui

            run_gui(config)
    except KeyboardInterrupt:
        log.info("Interrupt received; exiting")
    except ApplicationError as exc:
        log.error("Startup failed: %s", exc)


def _run_headless(config) -> None:
    app = Application(config)
    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    # Required for the 'spawn' start method used by process mode (esp. on Windows
    # and when frozen); harmless otherwise.
    multiprocessing.freeze_support()
    main()
