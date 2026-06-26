# PaintCam

PaintCam is a gesture-controlled camera pipeline. It reads a physical camera,
draws and zooms based on hand gestures, and writes the processed frames to a
virtual camera so other apps can use the output feed.

## Stack

- Tauri: desktop control panel and process lifecycle
- Python: camera capture, gesture recognition, drawing, zoom, and video output
- OpenCV: frame processing and preview
- MediaPipe: hand landmarks
- pyvirtualcam: virtual camera output

## Current gestures

- Point at the bottom palette to choose a color.
- Pinch thumb and index finger above the palette to draw.
- Use two visible hands; move index fingers apart to zoom in and closer together
  to zoom out.

## Run the Python engine directly

```sh
python3 -m pip install -r requirements.txt
python3 engine/paintcam_engine.py
```

Useful flags:

```sh
python3 engine/paintcam_engine.py --camera-index 1
python3 engine/paintcam_engine.py --no-virtual-camera
python3 engine/paintcam_engine.py --no-preview
```

On macOS, `pyvirtualcam` needs a virtual camera backend such as the OBS Virtual
Camera. If no backend is available, run with `--no-virtual-camera` while working
on gestures.

## Run the Tauri app

```sh
npm install
npm run tauri:dev
```

The Tauri app starts and stops `engine/paintcam_engine.py`. During development,
running the Python script directly is the fastest way to tune gestures.

## Adding gestures

Gestures live in `engine/paintcam/app.py` and implement the `Gesture` protocol:

```python
class MyGesture:
    name = "my_gesture"

    def update(self, context: FrameContext) -> None:
        ...
```

Register a gesture by adding it to `PaintCamEngine.gestures`. The gesture gets
the current frame, persistent drawing canvas, detected hands, selected color,
zoom level, and engine config through `FrameContext`.

The likely next refactor is moving each gesture into `engine/paintcam/gestures/`
once there are more than a handful.
