# PaintCam

PaintCam is a gesture-controlled camera pipeline. A Tauri control panel starts
and monitors a Python engine that uses OpenCV and MediaPipe, then optionally
writes the processed feed through pyvirtualcam.

## Stack and current behavior

- Tauri 2 / TypeScript / Vite: controls and engine status
- Rust: Python process lifecycle and JSONL event bridge
- Python / OpenCV / MediaPipe: mirrored capture, hand tracking, drawing, zoom,
  preview, and optional landmark overlay
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

Use a virtual environment when possible:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

OpenCV, NumPy, and MediaPipe are required. Startup prints a structured
`dependency_status` JSONL event and exits cleanly if one is unavailable.
pyvirtualcam is only required when virtual camera output is requested.

On macOS, pyvirtualcam also needs a backend such as OBS Virtual Camera. If the
package exists but no backend is available, PaintCam reports the problem and
continues with preview-only output. Use `--no-virtual-camera` to disable it
explicitly.

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
python3 engine/paintcam_engine.py --draw-landmarks --no-virtual-camera
python3 engine/paintcam_engine.py --debug-overlay --no-virtual-camera
python3 engine/paintcam_engine.py --draw-landmarks --debug-overlay --no-virtual-camera
python3 engine/paintcam_engine.py --brush-size 24 --no-virtual-camera
python3 engine/paintcam_engine.py \
  --gesture-config engine/gesture-config.example.json \
  --debug-overlay --no-virtual-camera
python3 engine/paintcam_engine.py --help
```

The debug overlay shows the winning gesture, confidence, selected color, brush
size, zoom, hand count, virtual-camera state, and any cooldown/conflict state.
Landmarks can be enabled at the same time when tuning detection geometry.

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
- `pinch_debounce_ms`: time a pinch must remain stable before drawing
- `palette_height_ratio`: bottom fraction of the frame reserved for the palette
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

The control panel sets camera index, preview, virtual camera, landmark overlay,
debug overlay, and brush size. It displays the active gesture and confidence
alongside process, camera, color, zoom, virtual-camera, conflict, error, and
recent JSONL/log state. Python is launched from the working source tree; full
sidecar packaging is intentionally not part of this milestone.

## Tests and validation

The Python helper tests do not access a camera:

```sh
npm run test:python
npm run build
cd src-tauri && cargo check
python3 -m compileall engine
```

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
- The Rust bridge starts `python3`; interpreter/environment selection is not yet
  configurable.
- Virtual camera availability depends on an OS-specific backend.
- Confidence is currently geometric confidence derived from normalized
  landmarks, not a trained per-gesture probability.
- Gesture thresholds vary with camera angle, hand size, lighting, and
  MediaPipe landmark jitter and may need local tuning.
- Stopping from Tauri terminates the child process, so the Python process may
  not have time to emit its final shutdown event; Rust still clears state.
- Gesture and camera behavior cannot be fully validated without real hardware.
