# Adaptive Face Detection — OpenVINO AUTO Plugin Backend

A production-style **backend / core engine** for continuous webcam face
detection built on the **OpenVINO AUTO Plugin**, with automatic device
discovery, **runtime device switching**, and **automatic fallback** across
**CPU, GPU and NPU**.

This is the headless core only — there is **no GUI**. The architecture is
deliberately event-driven so a Tkinter/Qt front-end can be layered on later by
subscribing to the application's events, without touching any core logic.

---

## Features

- 🎥 **Threaded webcam capture** (OpenCV) with a "latest-frame-wins" buffer for
  low latency.
- 🧠 **Continuous face detection** using the `face-detection-0200` OpenVINO model.
- 🧵 **Multi-threaded pipeline** — a capture thread and an inference thread run
  independently (plus monitor and reporter threads).
- ⚙️ **OpenVINO AUTO Plugin** for model compilation and inference.
- 🔎 **Automatic device discovery** (CPU / GPU / NPU).
- 🔀 **Runtime device switching** — change the active device without restarting.
- 🛟 **Automatic fallback** — recover from device errors or overload by migrating
  to another device.
- 📈 **Resource monitoring** (CPU, memory, and best-effort GPU) via `psutil`.
- 📝 **Structured logging** of FPS, inference time and the active device (no
  `print()`).

---

## Architecture

Each module has a single responsibility; `main.py` is only a thin entry point.

```
project/
├── main.py                 # Entry point only (picks front-end, graceful shutdown)
├── face_detection/         # The application package (all logic lives here)
│   ├── __init__.py         # Public API: Application, AppConfig, load_config
│   ├── app.py              # Composition root: wires components, owns lifecycle & events
│   ├── config.py           # Immutable, typed configuration + env-var loader
│   ├── logger.py           # Structured logging setup
│   ├── utils.py            # Generic helpers (FPSCounter, MovingAverage, Stopwatch)
│   ├── events.py           # Thread-safe observer/event bus (the UI seam)
│   ├── camera.py           # Camera wrapper + capture thread + latest-frame store
│   ├── device_manager.py   # OpenVINO Core, AUTO-plugin compilation, device switching
│   ├── inference.py        # FaceDetector: preprocess -> infer -> postprocess
│   ├── fallback.py         # FallbackPolicy (decision) + FallbackController (recovery)
│   ├── monitor.py          # ResourceMonitor thread + overload detection
│   ├── pipeline.py         # InferenceWorker thread + thread-safe PipelineStats
│   ├── worker.py           # Child-process inference entry point (process mode)
│   ├── supervisor.py       # Supervises the worker: crash + hang failover
│   ├── reporting.py        # ConsoleReporter (periodic structured stats line)
│   ├── visualization.py    # OpenCV frame-annotation helpers (boxes + overlay)
│   └── gui.py              # Live viewer (camera feed + device/FPS overlay)
├── models/                 # OpenVINO IR files (downloaded, not committed)
├── scripts/                # download_models.py
├── tests/                  # pytest unit tests
├── requirements.txt
├── pytest.ini
└── README.md
```

Only `main.py` sits at the root; every module lives in the `face_detection`
package and imports its siblings with relative imports, so the package is
self-contained and relocatable.

### Data flow

```
CameraThread ──put──> FrameStore ──get──> InferenceWorker ──> FaceDetector
                                                │                   │
                                                │ on inference error│
                                                ▼                   │
ResourceMonitor ──overload──> FallbackController ──switch──> DeviceManager
                                                │                   │ recompile
                                                └──> FaceDetector.set_compiled_model
```

All threads publish through the event bus; `Application` re-exposes them as
`on_result`, `on_device_change`, `on_resource`, and `on_fallback`.

---

## How the OpenVINO AUTO Plugin is used

The AUTO plugin is a **virtual device**. Instead of compiling for a fixed
device, we compile for a *candidate list* such as `AUTO:GPU,NPU,CPU`
(`DeviceManager.build_auto_string`). AUTO then:

1. Starts inference on CPU immediately (fast to load) and **transparently
   migrates** to the first higher-priority device (GPU, then NPU) once it is
   ready.
2. Provides **load-time fallback**: if a candidate device is missing or fails to
   load, AUTO silently selects the next one.

You can inspect what AUTO actually chose via
`DeviceManager.effective_device()`, which reads the `EXECUTION_DEVICES`
property of the compiled model.

## The two-layer fallback mechanism

| Layer | Handled by | Trigger | Action |
|-------|-----------|---------|--------|
| **Load-time** | OpenVINO AUTO plugin | Device missing / fails to compile | AUTO picks the next candidate |
| **Run-time** | `FallbackController` (this app) | Repeated inference errors **or** sustained resource overload | Quarantine the current device and **recompile** onto the next device per `FallbackPolicy` |

Run-time fallback works as follows:

1. The `InferenceWorker` catches an `InferenceError` and calls
   `FallbackController.handle_inference_error`; the `ResourceMonitor` calls
   `handle_overload` after several consecutive over-threshold samples.
2. Once the failure threshold is reached (or an overload is confirmed), the
   controller **quarantines** the current device (a cooldown so we don't bounce
   straight back onto it) and asks `FallbackPolicy.next_device` for the best
   remaining device, honouring the configured priority. CPU is the guaranteed
   last resort.
3. `DeviceManager.switch_device` recompiles the model onto the new device and
   `FaceDetector.set_compiled_model` swaps it in atomically — **no restart**.

A time-based **backoff** (`switch_backoff_sec`) gates how often a switch is even
attempted, so a device that fails on every frame at a high frame rate cannot
cause the controller to thrash or flood the log; per-frame failures are logged
at `DEBUG`, switch decisions at `INFO`.

The same `switch_device` path powers deliberate runtime switching via
`Application.switch_device("GPU")`.

---

## Installation

Requires **Python 3.9+** and a working OpenVINO runtime.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the model (into models/face-detection-0200/)
python scripts/download_models.py            # FP16 (recommended)
```

> GPU/NPU inference additionally requires the appropriate OpenVINO device
> plugins/drivers to be installed for your hardware. CPU always works.

---

## Usage

By default the app opens a live view: the camera feed with face boxes and a
minimal overlay of the **active device** and **FPS**.

```bash
python main.py           # GUI (default); press 'q' / Esc, or close the window, to quit
```

```
┌──────────────────────────────────┐
│ GPU.0  |  28.4 FPS                │   ← overlay drawn on the video
│                                  │
│        live webcam feed          │
│        with green face boxes     │
│                                  │
└──────────────────────────────────┘
```

The window is intentionally minimal — just the camera feed (an auto-sized
OpenCV window, no panels or menu bars) with the overlay drawn onto the frame.
Device selection is fully automatic (the **AUTO plugin**), so the overlay shows
whatever hardware AUTO is currently executing on; it may migrate CPU → GPU
shortly after start-up and the overlay updates live.

### Headless mode

To run without any window — pure backend with a periodic structured stats line
(`faces=N` is the live detection count) — use `--headless`:

```bash
python main.py --headless
```

### Process-isolation mode (survives a device hang/crash) — the default

Inference runs in an **isolated worker process** supervised by the parent by
default, so the application is immune to native device faults out of the box.
Add `--thread` to opt into in-process inference instead:

```bash
python main.py                        # GUI + isolated worker (default)
python main.py --headless             # headless + isolated worker
python main.py --thread               # in-process inference (lowest latency)
```

Why: a thread cannot recover from a native OpenVINO/driver **hang** or **crash**
(GPU reset, segfault) — it would freeze or take down the whole app. In process
mode the supervisor detects a dead worker (`proc.is_alive()`) or a hung one
(no result within `frame_timeout`), kills it, and **respawns it automatically**
with the crashed device family excluded (see *Resilience*); after
`max_restarts` it shuts down cleanly. The AUTO plugin still does its own
device selection/fallback *inside* the worker.

Trade-off of the default: slightly higher latency (frames cross a process
boundary) and no *manual* `switch_device()` API (overload-driven switching
still happens automatically inside the worker). Use `--thread` when you want
the lowest latency or interactive device switching and can accept that a
native device crash kills the app.

```
2026-07-01 18:02:11 | INFO | stats | device=GPU.0 | fps= 27.4 | infer=   4.7ms (avg   4.9ms) | faces=1 | frames=812 | errors=0 | cpu=20.3% mem=79.2% gpu=n/a
```

Press **Ctrl-C** to stop; either mode shuts every thread down gracefully and
releases the camera.

> The GUI (`gui.py`) is a thin presentation layer — it simply subscribes to
> `Application.on_result` (the same event any front-end would use) and changes
> **no core module**. `main.py` just picks the front-end; all detection logic
> lives in the backend regardless of mode.

### Configuration

All settings have sensible defaults and can be overridden with environment
variables (see `config.py`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `FD_CAMERA_INDEX` | `0` | Webcam index |
| `FD_CAMERA_WIDTH` / `FD_CAMERA_HEIGHT` | `640` / `480` | Capture resolution |
| `FD_CAMERA_DISCONNECT_GRACE` | `1.0` | Seconds of continuously failing reads before the camera is declared disconnected and rebuilt |
| `FD_CAMERA_RECONNECT_DELAY` / `FD_CAMERA_RECONNECT_MAX_DELAY` | `1` / `10` | Recovery backoff: initial delay between reopen attempts, doubling up to the cap (s) |
| `FD_CAMERA_HANG_TIMEOUT` | `5.0` | A read stuck in the driver longer than this is force-released to unblock the capture thread (`0` disables) |
| `FD_MODEL_XML` | `models/.../face-detection-0200.xml` | Path to the IR model |
| `FD_CONFIDENCE` | `0.5` | Minimum detection confidence |
| `FD_DEVICE_PRIORITY` | `GPU,NPU,CPU` | AUTO candidate + fallback order |
| `FD_PERF_HINT` | `LATENCY` | OpenVINO performance hint |
| `FD_ALLOW_GPU` / `FD_ALLOW_NPU` | `true` | Enable/disable a device family |
| `FD_MAX_FAILURES` | `3` | Inference errors before a device switch |
| `FD_COOLDOWN` | `30` | Seconds a failed device stays quarantined |
| `FD_SWITCH_BACKOFF` | `5` | Minimum seconds between switch attempts (anti-thrash) |
| `FD_PREFERRED_RETRY` | `60` | How often to check whether a recovered higher-priority device can take the work back (`0` disables) |
| `FD_MONITOR_INTERVAL` | `1.0` | Resource sampling interval (s) |
| `FD_CPU_THRESHOLD` / `FD_MEM_THRESHOLD` | `90` | System overload thresholds (%) |
| `FD_LATENCY_THRESHOLD` | `300` | Avg inference time (ms) above which the active device is deemed overloaded → switch (`0` disables) |
| `FD_INFERENCE_MODE` | `process` | `process` (isolated worker, crash-immune) or `thread` (same as `--thread`) |
| `FD_FRAME_TIMEOUT` | `3.0` | Process mode: hang timeout before respawning the worker (s) |
| `FD_MAX_RESTARTS` | `10` | Process mode: worker restarts (in quick succession) before giving up |
| `FD_CRASH_COOLDOWN` | `300` | Seconds a device family the worker died on stays excluded (doubles per repeat offence; a successful switch back rehabilitates it) |
| `FD_CACHE_DIR` | `.ov_cache` | OpenVINO kernel cache dir (`""` disables) |
| `FD_LOG_LEVEL` | `INFO` | Logging level |
| `FD_LOG_FILE` | *(none)* | Optional log file path |

Example:

```bash
FD_DEVICE_PRIORITY="NPU,GPU,CPU" FD_CONFIDENCE=0.6 python main.py
```

---

## GPU monitoring

`psutil` does not expose GPU utilisation, so the monitor sources it as follows:

- **Windows**: the vendor-neutral *GPU Engine* performance counters (what Task
  Manager's GPU graph reads) — no extra packages needed. Because OpenVINO's
  GPU plugin runs on **Intel silicon only**, the probe scopes itself to Intel
  adapters when any exist (adapter LUIDs from the counter instance names are
  matched against the `HKLM\SOFTWARE\Microsoft\DirectX` registry), so a busy
  discrete NVIDIA card can never masquerade as inference-GPU load. The busiest
  matching engine's utilisation is reported.
- **Elsewhere / fallback**: the optional `pynvml` package (NVIDIA only):
  `pip install nvidia-ml-py`.

If neither source is available, GPU load is reported as `n/a`, the
`FD_GPU_THRESHOLD`-based overload signal simply never fires, and everything
else continues to work.

---

## Extending with a UI

The backend never changes to add a UI — a front-end just subscribes to the
application's events and calls its public API. `face_detection/gui.py` is a
working example of exactly this pattern; the essentials are:

```python
from face_detection import Application, load_config

app = Application(load_config())

app.on_result.subscribe(lambda frame, detections, stats: ui.draw(frame, detections))
app.on_device_change.subscribe(lambda req, eff: ui.set_device_label(eff))
app.on_resource.subscribe(ui.update_gauges)
app.on_fallback.subscribe(lambda src, dst: ui.notify(f"Fell back {src} -> {dst}"))

# Runtime device switching from a button:
app.switch_device("GPU")

app.run()   # typically on a background thread, with the UI loop on the main thread
```

---

## Testing

```bash
pytest
```

The suite covers the pure logic that needs neither a camera nor a GPU:
configuration/env overrides, the event bus, the fallback policy and controller,
and the detector's SSD pre/post-processing.

---

## Notes / limitations

- NPU / GPU execution requires the matching OpenVINO device plugins and drivers;
  without them the AUTO plugin (and this app's fallback) gracefully use CPU.
- The model files are binary artefacts and are **not** committed — run the
  download script first.
- **Start-up speed:** on Windows the camera opens via DirectShow (much faster
  than the default MSMF backend, which can take several seconds to enumerate
  devices), the camera open and model compile run concurrently, and the GUI
  shows a window immediately with a "Starting…" placeholder. First frame
  typically appears within a few seconds of `python main.py`.

## Resilience

Three independent failure domains are handled separately:

- **Compute device — recoverable errors & overload (both modes).** In-process
  device fallback (`fallback.py`) recompiles onto another device at runtime when
  the active device is unhealthy. It triggers on: repeated **inference errors**;
  **system overload** (CPU/RAM past threshold); or, most usefully, **device
  overload** — the active device's average **inference latency** exceeding
  `FD_LATENCY_THRESHOLD` (a slow/degraded device switches even if it never throws
  and system RAM looks fine). A quarantine + backoff prevents thrashing. The
  AUTO plugin adds its own load-time device selection/fallback. Switches are
  near-instant: a CPU standby is pre-compiled in the background at start-up and
  every concrete-device compile is cached, so fleeing a failing device (or
  moving off an overloaded CPU onto the GPU, and back) needs no on-crisis
  compilation. Errors that indicate the device itself died (driver reset,
  `DEVICE_LOST`, context lost) switch on the *first* failure instead of waiting
  for the threshold, and a repeatedly-erroring device gets an escalating
  quarantine so an overload signal cannot bounce inference straight back onto
  a broken accelerator. Overload handling is **resource-aware**: a switch only
  happens when the saturated resource is the one running inference — a CPU
  breach while inference runs on the GPU is somebody else's workload and is
  ignored, and a memory breach alone never migrates (switching devices frees
  no RAM). In process mode the switch is delivered to the worker as a
  ``("switch", device)`` control message, so overload-driven switching works
  identically in both modes.
- **Compute device — hangs & crashes (process mode, the default).** A native
  driver hang or crash (GPU reset, segfault) can't be recovered by a thread. Here
  the supervisor (`supervisor.py`) detects a dead or hung worker and **respawns
  it automatically**; after `max_restarts` *in quick succession* it shuts down
  cleanly (the budget resets after a healthy minute, so isolated failures over
  hours never exhaust it). The respawned worker **excludes the device family
  the previous worker died on** — a worker that crashed while executing on GPU
  comes back on CPU within ~2 seconds instead of crash-looping. The exclusion
  is **timed and escalating** (`FD_CRASH_COOLDOWN`, default 300 s, doubling per
  repeat offence): a device wrongly blamed for a hang while the whole machine
  was starved gets another chance after the cooldown, and a successful switch
  back onto it fully rehabilitates it. CPU itself is never excluded. The
  return does **not** wait for an overload: a periodic preferred-device check
  (`FD_PREFERRED_RETRY`, default 60 s, both modes) moves inference back onto
  the recovered accelerator as soon as its quarantine expires, even when the
  CPU is coping fine.
- **Camera.** Recovery is fully automatic and confined to the capture thread —
  no dialogs, no buttons, the GUI just keeps running. All failure shapes are
  handled: reads returning no frame, driver exceptions (MSMF's `cv2.error`),
  and even a read *hung inside the driver* (a watchdog force-releases the
  device to unblock it). Once reads have failed for longer than a short grace
  period the dead `VideoCapture` is released and recreated from scratch —
  DirectShow first, then MSMF — with resolution/FPS settings restored and
  exponential backoff between attempts, so a camera that is absent for minutes
  costs nothing but is picked up within seconds of replugging. Streaming and
  inference resume on their own; during an outage the reported FPS drops to
  `0` instead of freezing at the last value.

> Note: on most laptops, disabling the GPU in Device Manager also disables the
> integrated webcam, so that emulates a **camera** outage, not a compute-device
> crash. To exercise compute fallback specifically, induce an inference error or
> overload on the active device (the camera must keep delivering frames for the
> inference path to run and detect the failure).
