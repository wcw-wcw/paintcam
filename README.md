# PaintCam

PaintCam is a gesture-controlled camera pipeline. A Tauri control panel starts
and monitors a Python engine that uses OpenCV and MediaPipe, then optionally
writes the processed feed through pyvirtualcam.

## Stack and current behavior

- Tauri 2 / TypeScript / Vite: controls and engine status
- Rust: Python process lifecycle and JSONL event bridge
- Python / OpenCV / MediaPipe Tasks Hand Landmarker: mirrored capture, hand
  tracking, drawing, zoom, preview, and optional landmark overlay
- pyvirtualcam: optional virtual camera output

Current gestures:

- Point at a bottom palette swatch to choose a color.
- Hold a thumb/index pinch briefly above the palette to draw. A separate release
  threshold prevents noisy landmark measurements from flickering the stroke.
- Show two hands and move their index fingers apart or together to zoom.

The processed preview is mirrored. Hand landmarks are available with
`--draw-landmarks`; live gesture diagnostics are available with
`--debug-overlay`. They are independent and off by default. Palette selection
has priority over zoom, and two-hand zoom has priority over drawing.

## Python dependencies

Create the repo-local virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

OpenCV, NumPy, and the current MediaPipe Tasks API are required. PaintCam uses
`vision.HandLandmarker` in synchronous video mode; it does not use the legacy
`mp.solutions` API. Startup prints a structured `dependency_status` JSONL event
and exits cleanly if an import or the required Hand Landmarker API is unavailable.
pyvirtualcam is only required when virtual camera output is requested.

On macOS, pyvirtualcam also needs a backend such as OBS Virtual Camera. If the
package exists but no backend is available, PaintCam reports the problem and
continues with preview-only output. Use `--no-virtual-camera` to disable it
explicitly.

## Hand Landmarker model

MediaPipe Tasks requires a separate `.task` model bundle. Download Google's
Hand Landmarker model into the default location:

```sh
python3 scripts/download-mediapipe-models.py
ls -lh engine/models/hand_landmarker.task
```

The downloaded model is intentionally ignored by Git and source archives. To
use a model stored elsewhere:

```sh
python3 engine/paintcam_engine.py \
  --hand-model /path/to/hand_landmarker.task \
  --no-virtual-camera
```

If the model is absent, startup emits an `error` JSONL event with code
`hand_landmarker_model_missing` and the resolved expected path.

Check the environment and default model without opening a camera:

```sh
python3 engine/paintcam_engine.py --doctor
```

The Tauri control panel also has a **Run doctor** button. The doctor event
reports the resolved Python executable, Python, OpenCV, NumPy, and MediaPipe versions; the
MediaPipe install path; Tasks HandLandmarker availability; default model path
and existence; and pyvirtualcam availability.

## Run the engine directly

Development without a virtual camera:

```sh
python3 engine/paintcam_engine.py --no-virtual-camera
```

Headless development:

```sh
python3 engine/paintcam_engine.py --no-preview --no-virtual-camera
```

Other useful options:

```sh
python3 engine/paintcam_engine.py --camera-index 1 --no-virtual-camera
python3 engine/paintcam_engine.py --hand-model ./models/custom.task --no-virtual-camera
python3 engine/paintcam_engine.py --draw-landmarks --no-virtual-camera
python3 engine/paintcam_engine.py --debug-overlay --no-virtual-camera
python3 engine/paintcam_engine.py --draw-landmarks --debug-overlay --no-virtual-camera
python3 engine/paintcam_engine.py --brush-size 24 --no-virtual-camera
python3 engine/paintcam_engine.py \
  --gesture-config engine/gesture-config.example.json \
  --debug-overlay --no-virtual-camera
python3 engine/paintcam_engine.py --help
python3 engine/paintcam_engine.py --list-cameras
python3 engine/paintcam_engine.py --virtual-camera-probe
```

`--list-cameras` probes indexes 0–4 and reports which devices open and return a
frame. It needs OpenCV, but does not import/use HandLandmarker and does not
require the `.task` model. The same probe is available in the Tauri UI.

The debug overlay shows the winning gesture, confidence, selected color, brush
size, zoom, hand count, virtual-camera state, and any cooldown/conflict state.
Landmarks can be enabled at the same time when tuning detection geometry.
These diagnostic overlays appear in the local preview only by default. Pass
`--virtual-camera-overlays` (or enable **Include diagnostics in virtual
output**) to send them to the virtual camera too.

Press `q` in the preview window to stop. Stdout is reserved for one JSON object
per line. Events include engine/dependency lifecycle, camera frames, gesture
state, virtual camera state, errors, and shutdown. Human-readable diagnostics
are written to stderr. Gesture events are change-aware and throttled to at most
five per second, with a one-second heartbeat.

## Gesture tuning

Defaults live in `GestureTuning` in `engine/paintcam/gestures.py`. Copy
`engine/gesture-config.example.json`, change only the values you want to tune,
and pass it with `--gesture-config`.

Available settings:

- `pinch_distance_threshold` and `pinch_release_threshold`: normalized
  thumb/index activation and hysteresis distances
- `pinch_debounce_ms` and `pinch_min_stable_frames`: time and consecutive
  frames a pinch must remain stable before drawing
- `draw_deadzone_px`: minimum fingertip movement before another stroke segment
  is added
- `palette_height_ratio`: bottom fraction of the frame reserved for the palette
- `palette_hold_ms` and `palette_min_stable_frames`: confirmation required
  before a hovered swatch becomes active
- `palette_cooldown_ms`, `draw_cooldown_ms`, and `zoom_cooldown_ms`
- `zoom_sensitivity`, `min_zoom`, and `max_zoom`
- `brush_size`: output stroke width in pixels

Unknown keys and invalid ranges produce a structured startup error. The
`--brush-size` CLI option overrides the configured brush size.

## Run the Tauri app

```sh
npm install
npm run tauri:dev
```

`npm run tauri:dev` performs a fresh frontend build before every development
launch. Tauri then lets Cargo rebuild any changed Rust code and starts the Vite
development server, so no separate manual build step is needed.

You do not need to activate `.venv` before launching Tauri. The Rust bridge
resolves Python in this order:

1. The optional Python executable entered in the control panel.
2. Repo-local `.venv/bin/python`, when present.
3. `python3` found on the shell `PATH` inherited by Tauri.
4. A conventional system Python fallback, only if necessary.

The resolved executable is displayed in engine state and doctor output. An
explicit override is saved locally by the frontend; leave it blank for
automatic resolution. The control panel sets camera index, preview, virtual camera, landmark overlay,
debug overlay, and brush size. It displays the active gesture and confidence
alongside process, camera, color, zoom, virtual-camera, conflict, error, and
recent JSONL/log state. Python is launched from the working source tree; full
sidecar packaging is intentionally not part of this milestone.

While the engine is running, **Clear canvas**, **Reset zoom**, **Pause/Resume
drawing**, and brush size changes are sent over a bounded JSONL stdin command
channel. Preview and hand tracking continue while drawing is paused. Gesture
threshold changes in a config file require an engine restart.

## Virtual camera testing

PaintCam sends a mirrored, resized camera frame composited with the persistent
drawing canvas, palette, and PaintCam drawing status. Landmark and debug
diagnostics are preview-only unless explicitly included with
`--virtual-camera-overlays`. The frame passed to pyvirtualcam is always resized
to the virtual camera's declared width and height.

Probe support without opening the physical camera or loading MediaPipe:

```sh
python3 engine/paintcam_engine.py --virtual-camera-probe
```

The probe exits after emitting JSONL describing whether pyvirtualcam imported,
whether a test camera was created, the selected backend when reported by
pyvirtualcam, output dimensions/FPS, and any error. The same check is available
through **Probe virtual camera** in the control panel.

OBS-first workflow:

1. Run the probe and confirm `created` is `true`.
2. Start PaintCam with preview and virtual camera enabled.
3. Confirm the UI reports **active**, a backend, output size, advancing virtual
   frame count, and a recent write time.
4. In OBS, add a Video Capture Device source and select the virtual camera
   exposed by the installed backend.
5. Compare OBS with preview for mirroring, drawing, palette, output dimensions,
   and steady FPS. Diagnostic overlays should appear only where configured.
6. Only after OBS works, test the camera selector in Zoom, Discord, or another
   consumer. Restart that app if it was open before the backend became
   available.

Common failures:

- **pyvirtualcam missing:** install the project requirements in the Python
  executable shown by PaintCam, then rerun the probe.
- **No compatible backend:** install or start a backend supported by
  pyvirtualcam on the current platform. Backend names and setup vary, so use
  the probe error and the backend's own documentation.
- **Consumer does not show the camera:** prove the feed in OBS first, then
  restart the consumer and recheck its camera permissions and device list.
- **Camera already in use:** close other camera producers/consumers and retry
  both the physical-camera and virtual-camera probes.
- **Low FPS:** compare capture FPS and virtual output FPS, close competing
  video apps, disable preview diagnostics, and test a lower width, height, or
  requested FPS from the CLI.

Virtual-camera lifecycle events use `disabled`, `initializing`, `active`,
`unavailable`, and `failed`. Once active, throttled events report backend,
size, requested/output FPS, frame count, last successful write time, and write
failure count. A backend creation or write failure does not prevent the local
preview from continuing.

## First successful camera test

1. Download the hand model and run **Run doctor**; confirm OpenCV, MediaPipe,
   HandLandmarker, and the model all report available.
2. Run **Probe cameras 0–4** and select an index reported open and readable.
3. Disable virtual camera initially, enable preview, landmarks, and debug
   overlay, then start the engine.
4. Confirm Camera open is **Yes**, FPS/frame count advance, and landmarks follow
   one hand.
5. Hold the index fingertip over a palette cell until its active outline moves.
6. Pinch thumb/index above the palette and hold briefly; move steadily to draw.
7. Pause drawing and verify preview/tracking continue without new strokes.
8. Clear the canvas, reset zoom, then use two hands to verify zoom owns the
   frame and blocks drawing.
9. Enable virtual camera only after the preview path works.

## Live tuning guidance

- Drawing triggers too easily: lower `pinch_distance_threshold`, increase
  `pinch_debounce_ms` or `pinch_min_stable_frames`, and restart.
- Drawing does not trigger: raise `pinch_distance_threshold` slightly, verify
  thumb/index landmarks are stable, or reduce debounce/stable frames.
- Palette flickers: increase `palette_hold_ms` or
  `palette_min_stable_frames`. Drawing is blocked immediately anywhere in the
  palette region, even before selection confirms.
- Zoom is too sensitive: lower `zoom_sensitivity`.
- Landmarks lag: improve lighting, keep the hand fully visible, close other
  camera consumers, and compare measured FPS with the requested FPS.
- FPS is low: reduce `--width`, `--height`, or `--fps` when running directly;
  disable landmarks/debug overlay and virtual camera while isolating the
  bottleneck.

## Tests and validation

The Python helper tests do not access a camera:

```sh
npm run test:python
npm run build
cd src-tauri && cargo check
python3 -m compileall engine
cargo test --manifest-path src-tauri/Cargo.toml
```

### Common startup errors

- Wrong Python executable: inspect **Python executable** in engine state or run
  doctor. Clear the UI override to restore automatic resolution, or enter the
  desired interpreter path explicitly.
- MediaPipe missing from system Python: create/install the repo-local `.venv`;
  the bridge will prefer it automatically. Doctor identifies the interpreter
  and reports whether MediaPipe and HandLandmarker are available.
- Missing `.task` model: the model is intentionally not committed. Run
  `python3 scripts/download-mediapipe-models.py`, or
  pass its location with `--hand-model`.
- Camera permission does not prompt: run the UI camera probe or direct
  `--list-cameras`, then allow camera access for Terminal/the Tauri development
  host in the operating system privacy settings and restart the app.
- Camera index unavailable: close other camera users, use **Probe cameras
  0–4**, and select an index reported as open/readable.
- Virtual camera unavailable: install/start an OS backend such as OBS Virtual
  Camera, or test with `--no-virtual-camera`.

### Suggested local camera checklist

1. Start with `--draw-landmarks --debug-overlay --no-virtual-camera`.
2. Confirm the mirrored landmarks follow one hand without obvious swaps.
3. Point into every palette swatch and verify drawing never starts there.
4. Hold a pinch above the palette; verify drawing starts after the short
   debounce and stays active until the wider release threshold.
5. Add a second hand during drawing; verify zoom wins immediately and no stroke
   is added.
6. Move both index fingertips together/apart and check zoom continuity and
   configured clamps.
7. Restart with virtual camera enabled and verify its feed in a second app.
8. Repeat in dim and bright lighting and tune normalized thresholds if needed.

## Source-only archives

Create a shareable source archive with:

```sh
./scripts/export-source-zip.sh
```

You may pass a different output filename as the first argument. Source zips
should exclude generated, dependency, cache, and VCS content including
`src-tauri/target`, `node_modules`, `dist`, `build`, `.git`, `.venv`,
`__pycache__`, `__MACOSX`, and `.DS_Store`. These folders are reproducible,
large, or machine-specific and should not be committed or shared as source.

## Adding gestures

Gesture decisions live in `engine/paintcam/gestures.py` and do not import
OpenCV or MediaPipe. A gesture declares `name`, `priority`, `cooldown_ms`,
`confidence_threshold`, and an optional `exclusive_group`, then returns a
`GestureResult` from `evaluate`. Register it in `default_registry`.

`GestureContext` supplies frame dimensions, normalized hand landmarks, palette,
drawing and zoom state, tuning values, timestamp, and frame index. The engine
applies the winning result's action to the canvas or state. Keeping evaluation
camera-free makes conflict and timing behavior unit-testable.

## Known limitations

- Python and its dependencies must currently be installed separately.
- Python is resolved at development runtime; packaged sidecar handling is
  intentionally deferred.
- Virtual camera availability depends on an OS-specific backend.
- Confidence is currently geometric confidence derived from normalized
  landmarks, not a trained per-gesture probability.
- Gesture thresholds vary with camera angle, hand size, lighting, and
  MediaPipe landmark jitter and may need local tuning.
- Stopping from Tauri terminates the child process, so the Python process may
  not have time to emit its final shutdown event; Rust still clears state.
- Gesture and camera behavior cannot be fully validated without real hardware.
