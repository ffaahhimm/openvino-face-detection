"""Application configuration.

All tunable parameters live here as immutable, strongly-typed dataclasses.
Configuration is loaded once at start-up (see :func:`load_config`) and then
passed down to every component via dependency injection, so no module ever
reaches for a global or reads the environment on its own.

Environment variables (all optional) override the built-in defaults, which
keeps the code twelve-factor friendly without pulling in a heavyweight config
framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Small env-parsing helpers (kept private to this module)
# ---------------------------------------------------------------------------


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw is not None else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = os.environ.get(key)
    if not raw:
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraConfig:
    """Webcam capture settings."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    # Frames to grab and discard on start-up so auto-exposure settles. Kept small
    # so it does not slow the launch; the live feed settles within a moment anyway.
    warmup_frames: int = 2
    # Pause between retries when reads are failing, so a fast-failing device
    # (unplugged, reset, permission lost, ...) cannot spin the CPU or flood logs.
    error_retry_delay_sec: float = 0.2
    # Reads failing continuously for this long (or a device that closed outright)
    # mean the camera is gone: the capture thread declares it disconnected and
    # recreates the VideoCapture from scratch. Short, so outages are noticed
    # quickly; brief single-frame glitches are retried within the grace period.
    disconnect_grace_sec: float = 1.0
    # Initial delay between reopen attempts during recovery; doubles per failed
    # attempt (exponential backoff) up to reconnect_max_delay_sec.
    reconnect_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 10.0
    # A single read() blocked inside the driver for longer than this is treated
    # as a hang: the watchdog force-releases the device to unblock the capture
    # thread, which then recovers normally. 0 disables the watchdog.
    read_hang_timeout_sec: float = 5.0


@dataclass(frozen=True)
class ModelConfig:
    """Face-detection model settings (``face-detection-0200`` by default)."""

    xml_path: str = os.path.join(
        "models", "face-detection-0200", "face-detection-0200.xml"
    )
    # ``.bin`` is inferred from the ``.xml`` when left empty.
    bin_path: str = ""
    # Detections below this confidence are discarded.
    confidence_threshold: float = 0.5


@dataclass(frozen=True)
class DeviceConfig:
    """OpenVINO device / AUTO-plugin settings.

    ``auto_priority`` is the ordered candidate list handed to the AUTO plugin,
    e.g. ``AUTO:GPU,NPU,CPU`` -- AUTO picks the first device that can load the
    model and transparently falls back to the next one.
    """

    # Preference order used both for the AUTO candidate string and for our own
    # application-level fallback policy.
    auto_priority: List[str] = field(default_factory=lambda: ["GPU", "NPU", "CPU"])
    # OpenVINO performance hint: "LATENCY" (best for real-time) or "THROUGHPUT".
    performance_hint: str = "LATENCY"
    # If False, NPU is never offered to AUTO nor used as a fallback target.
    allow_npu: bool = True
    # If False, GPU is excluded entirely.
    allow_gpu: bool = True
    # OpenVINO on-disk kernel cache. Speeds up (re)compilation -- especially on
    # GPU and after a worker restart in process mode. Empty disables caching.
    cache_dir: str = ".ov_cache"


@dataclass(frozen=True)
class MonitorConfig:
    """Resource-monitoring thresholds (percentages)."""

    interval_sec: float = 1.0
    cpu_threshold: float = 90.0
    memory_threshold: float = 90.0
    gpu_threshold: float = 95.0
    # Average inference time (ms) above which the *active device* is considered
    # overloaded/degraded and a fallback switch is triggered. Set well above a
    # healthy device's latency so it only fires on genuine slowdowns; 0 disables.
    inference_latency_threshold_ms: float = 300.0
    # Consecutive over-threshold samples required before an overload is signalled.
    overload_samples: int = 3


@dataclass(frozen=True)
class FallbackConfig:
    """Device-recovery / fallback behaviour."""

    # Consecutive inference errors on the active device before we switch away.
    max_consecutive_failures: int = 3
    # A device that just failed is quarantined for this long before reuse.
    cooldown_sec: float = 30.0
    # Minimum time between switch *attempts*. Prevents thrashing / log floods
    # when a device fails on (almost) every frame at a high frame rate.
    switch_backoff_sec: float = 5.0
    # Whether an overload signal (not just an error) should trigger a switch.
    switch_on_overload: bool = True
    # How often to check whether a higher-priority device has become healthy
    # again and move inference back onto it. This is what returns work to a
    # recovered GPU even when nothing is overloaded; 0 disables the check.
    preferred_retry_sec: float = 60.0


@dataclass(frozen=True)
class InferenceConfig:
    """How inference is executed.

    ``process`` (default): inference runs in an isolated child process
    supervised by the parent. A device *hang* or *crash* (driver fault, GPU
    reset, segfault) is then contained and the worker is respawned
    automatically on the remaining healthy devices -- full fault immunity,
    which is why it is the default.

    ``thread``: inference runs in a thread in this process -- lowest latency
    and runtime ``switch_device()``, but a native device crash takes down the
    whole application.
    """

    mode: str = "process"  # "process" | "thread"
    # Seconds to wait for a freshly spawned worker to report ready.
    worker_ready_timeout_sec: float = 30.0
    # If a submitted frame produces no result within this long, the worker is
    # considered hung: it is killed and respawned.
    frame_timeout_sec: float = 3.0
    # Pause before respawning a crashed/hung worker.
    restart_delay_sec: float = 0.5
    # Give up (and shut down) after this many restarts *in quick succession*.
    # The budget guards against crash-loops, not against a long uptime: once a
    # worker has stayed healthy for restart_budget_reset_sec, the count resets,
    # so isolated failures spread over hours can never exhaust it.
    max_restarts: int = 10
    restart_budget_reset_sec: float = 60.0
    # A device family the worker crashed/hung on is excluded for this long
    # (doubling per repeat offence, capped at 8x) rather than forever: one
    # false alarm -- e.g. a hang blamed on the GPU while the whole machine was
    # starved -- must not degrade the app to CPU for the rest of a long run.
    # After expiry an overload may offer the device again; a successful switch
    # back fully re-enables it.
    crash_cooldown_sec: float = 60.0


@dataclass(frozen=True)
class LoggingConfig:
    """Structured-logging configuration."""

    level: str = "INFO"
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    # Optional log file; when empty, logs go to the console only.
    file_path: str = ""


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration aggregating every section."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # How often (seconds) the console reporter prints a stats line.
    report_interval_sec: float = 2.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config() -> AppConfig:
    """Build an :class:`AppConfig`, letting environment variables override defaults.

    This is the single entry point used by ``main.py``. Keeping the override
    logic here means the rest of the codebase only ever sees a fully-resolved,
    immutable config object.
    """

    camera = CameraConfig(
        index=_env_int("FD_CAMERA_INDEX", CameraConfig.index),
        width=_env_int("FD_CAMERA_WIDTH", CameraConfig.width),
        height=_env_int("FD_CAMERA_HEIGHT", CameraConfig.height),
        fps=_env_int("FD_CAMERA_FPS", CameraConfig.fps),
        disconnect_grace_sec=_env_float(
            "FD_CAMERA_DISCONNECT_GRACE", CameraConfig.disconnect_grace_sec
        ),
        reconnect_delay_sec=_env_float(
            "FD_CAMERA_RECONNECT_DELAY", CameraConfig.reconnect_delay_sec
        ),
        reconnect_max_delay_sec=_env_float(
            "FD_CAMERA_RECONNECT_MAX_DELAY", CameraConfig.reconnect_max_delay_sec
        ),
        read_hang_timeout_sec=_env_float(
            "FD_CAMERA_HANG_TIMEOUT", CameraConfig.read_hang_timeout_sec
        ),
    )

    model = ModelConfig(
        xml_path=_env_str("FD_MODEL_XML", ModelConfig.xml_path),
        bin_path=_env_str("FD_MODEL_BIN", ModelConfig.bin_path),
        confidence_threshold=_env_float(
            "FD_CONFIDENCE", ModelConfig.confidence_threshold
        ),
    )

    device = DeviceConfig(
        auto_priority=_env_list("FD_DEVICE_PRIORITY", DeviceConfig().auto_priority),
        performance_hint=_env_str("FD_PERF_HINT", DeviceConfig.performance_hint),
        allow_npu=_env_bool("FD_ALLOW_NPU", DeviceConfig.allow_npu),
        allow_gpu=_env_bool("FD_ALLOW_GPU", DeviceConfig.allow_gpu),
        cache_dir=_env_str("FD_CACHE_DIR", DeviceConfig.cache_dir),
    )

    inference = InferenceConfig(
        mode=_env_str("FD_INFERENCE_MODE", InferenceConfig.mode),
        frame_timeout_sec=_env_float(
            "FD_FRAME_TIMEOUT", InferenceConfig.frame_timeout_sec
        ),
        max_restarts=_env_int("FD_MAX_RESTARTS", InferenceConfig.max_restarts),
        crash_cooldown_sec=_env_float(
            "FD_CRASH_COOLDOWN", InferenceConfig.crash_cooldown_sec
        ),
    )

    monitor = MonitorConfig(
        interval_sec=_env_float("FD_MONITOR_INTERVAL", MonitorConfig.interval_sec),
        cpu_threshold=_env_float("FD_CPU_THRESHOLD", MonitorConfig.cpu_threshold),
        memory_threshold=_env_float("FD_MEM_THRESHOLD", MonitorConfig.memory_threshold),
        inference_latency_threshold_ms=_env_float(
            "FD_LATENCY_THRESHOLD", MonitorConfig.inference_latency_threshold_ms
        ),
    )

    fallback = FallbackConfig(
        max_consecutive_failures=_env_int(
            "FD_MAX_FAILURES", FallbackConfig.max_consecutive_failures
        ),
        cooldown_sec=_env_float("FD_COOLDOWN", FallbackConfig.cooldown_sec),
        switch_backoff_sec=_env_float(
            "FD_SWITCH_BACKOFF", FallbackConfig.switch_backoff_sec
        ),
        preferred_retry_sec=_env_float(
            "FD_PREFERRED_RETRY", FallbackConfig.preferred_retry_sec
        ),
    )

    logging_cfg = LoggingConfig(
        level=_env_str("FD_LOG_LEVEL", LoggingConfig.level),
        file_path=_env_str("FD_LOG_FILE", LoggingConfig.file_path),
    )

    return AppConfig(
        camera=camera,
        model=model,
        device=device,
        monitor=monitor,
        fallback=fallback,
        inference=inference,
        logging=logging_cfg,
        report_interval_sec=_env_float(
            "FD_REPORT_INTERVAL", AppConfig.report_interval_sec
        ),
    )
