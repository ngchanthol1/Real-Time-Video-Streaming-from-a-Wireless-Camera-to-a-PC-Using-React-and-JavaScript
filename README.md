# RoboCam_75C0 — Face Recognition (React, matches chapter8.py)

A React app that connects to your RoboCam_75C0's WiFi video stream and
runs face **detection, landmarks, and named recognition** — using the
**same `.tflite` neural network models** that `RoboCam`'s Python library
uses internally. This is a from-scratch JavaScript port of the math in
`face_detector.py`, `face_landmark.py`, and `face_recognizer.py`, so the
boxes, landmarks, and name-matching behavior closely track what
`chapter8.py` does — not a generic substitute library.

It replicates chapter8.py's full menu:
- **1) Capture** — save a named face sample (with a 3-2-1 countdown, like the original)
- **2) Train** — re-detect faces in all saved samples and build recognition embeddings
- **3) Delete** — remove a person's trained data

No Python is required at runtime — everything (camera stream decode,
detection, landmarks, recognition, and storage of captured faces) runs
client-side in the browser.

---

## How it works, and how it differs from chapter8.py

| chapter8.py (Python) | This app (React/JS) |
|---|---|
| `urllib` reads `192.168.4.1:81/stream`, `cv2.imdecode` parses JPEG frames | `<img>` tag + `<canvas>` reads the same MJPEG stream natively in-browser |
| `face_detector.tflite` via `tf.lite.Interpreter` | Same `.tflite` file, run via `@tensorflow/tfjs-tflite`'s in-browser WASM interpreter |
| `face_keypoints.tflite` for 68-point landmarks | Same `.tflite` file, same model |
| `face_recognizer.tflite` to 192-dim embeddings, Euclidean distance, threshold `0.1` | Same model, same distance math, same `0.1` threshold |
| Saved faces as `.jpg` files in `res/face/` | Saved faces as data URLs in the browser's IndexedDB |
| Console `input()` menu (capture/train/delete) | Three on-screen panels doing the same three actions |

The detection/landmark/recognition **code itself is a manual line-by-line
port** of the Python preprocessing (letterboxing, NMS, square-crop-and-pad
math) — see `src/lib/faceDetectorUtils.js`, `src/lib/faceCropUtils.js`,
and `src/lib/RoboCamEngine.js` for the ported logic, with comments
pointing back to the original Python functions they correspond to.

---

## Step-by-step setup (Windows 11)

### Step 1 — Place the project

Unzip this project somewhere like:

```
C:\Workspace_2\Rogic\robocam-face-app
```

### Step 2 — Install dependencies (while on normal internet)

This app loads the TensorFlow.js libraries from a CDN at runtime (see
**Why CDN scripts?** below), so do this step on your regular WiFi/internet
connection, before switching to the camera's network:

```cmd
cd C:\Workspace_2\Rogic\robocam-face-app
npm install
```

This was tested with your installed versions — Node v16.20.0, npm
8.19.4 — using Vite 4.x (Vite 5+ requires Node 18+, so this project
intentionally avoids it).

### Step 3 — Connect your PC to the RoboCam_75C0's WiFi

1. Power on the camera.
2. Click the WiFi icon in Windows 11's system tray.
3. Connect to the camera's own network (check the camera's label/manual
   for the exact SSID/password).
4. Your PC will lose normal internet while connected to the camera's
   isolated hotspot — that's expected.

> Quick sanity check: open a plain browser tab to
> `http://192.168.4.1:81/stream` first. If you don't see video there,
> fix that before touching the React app — it can't do better than your
> browser can at the network level.

### Step 4 — Start the app

```cmd
npm run dev
```

```
VITE v4.5.x  ready in xxx ms
➜  Local:   http://localhost:5173/
```

### Step 5 — Open it in your browser

Go to `http://localhost:5173`. This works fine even while your WiFi is
on the camera's network, since `localhost` doesn't need internet — but
**the TFJS library scripts must have already loaded once** (they're
cached by the browser after first load). If this is your very first run,
load the page once *before* switching to the camera's WiFi so the CDN
scripts can download, then reconnect to the camera's WiFi afterward and
refresh.

### Step 6 — Use it

1. Wait for **Models: All models loaded.** in the header.
2. Confirm/edit the stream URL (defaults to `http://192.168.4.1:81/stream`).
3. Click **Connect to Camera**, then **Start Detection**.
4. Boxes, 68-point landmarks, and a name (`Human` until trained) appear
   over detected faces.
5. To register someone: type a name in the **Capture** box, click
   **Capture Face**, wait for the 3-2-1 countdown, then look at the
   camera. Repeat a few times per person for better accuracy (matching
   how `chapter8.py` works — more samples means more reference
   embeddings).
6. Click **Train Face Data** to build embeddings from everything
   captured so far.
7. From then on, recognized faces show their name instead of `Human`.
8. To remove someone, type their name under **Delete** and click
   **Delete Face Data**.

---

## Why CDN `<script>` tags instead of `npm install`-ing tfjs?

`@tensorflow/tfjs-tflite`'s npm package ships a broken build for modern
bundlers (Vite/Rollup/Webpack) — it references an internal WASM-loader
module in a way that only works when loaded as a classic browser
`<script>` tag, which is also TensorFlow's own documented usage pattern
for this specific library. Rather than fight the bundler with workarounds
that could break in subtle ways, this project loads the TFJS stack from
jsDelivr's CDN in `index.html`, exactly as TensorFlow's own examples do.

**Practically:** your PC needs internet access the *first* time it loads
the page (to fetch and cache these scripts), but not afterward — and the
neural network model files themselves (the `.tflite` files, which is the
RoboCam-specific part) are bundled locally in `public/models/`, not
fetched remotely.

---

## Project structure

```
robocam-face-app/
├── index.html                      ← loads TFJS via CDN scripts
├── package.json
├── vite.config.js
├── public/
│   └── models/
│       ├── face_detector.tflite     ← copied from RoboCam's Python package
│       ├── face_keypoints.tflite    ← (landmarks)
│       └── face_recognizer.tflite   ← (192-dim embeddings)
└── src/
    ├── main.jsx                     ← entry point, sets tfjs backend
    ├── App.jsx                      ← UI: stream, detection loop, capture/train/delete
    ├── App.css
    ├── index.css
    └── lib/
        ├── RoboCamEngine.js         ← ports robocam.py's model-wrapping logic
        ├── faceDetectorUtils.js     ← ports face_detector.py (letterbox, NMS)
        ├── faceCropUtils.js         ← ports the shared crop/embedding-distance math
        └── faceStorage.js           ← IndexedDB, replaces the res/face/ folder
```

---

## Troubleshooting

### "tfjs failed to load from CDN"
Your PC needs internet access at least once to fetch the TFJS scripts.
Load the page on your normal WiFi first, then switch to the camera's
network.

### "Could not reach http://192.168.4.1:81/stream"
- Confirm your PC's WiFi is connected to the **camera's** network.
- Test the URL directly in a plain browser tab first.
- Check the camera's documentation if it uses a different default IP.

### Detection runs but boxes are misplaced / landmarks look wrong
This usually means a coordinate-space mismatch between the detector's
letterboxed output and the canvas it's drawn on. Open the browser
console (`F12`) and check for errors during `detectFaces` — if frames
are being skipped (see console warnings), the camera resolution may be
changing mid-stream; try reconnecting.

### Recognition always says "Human", never a trained name
- Make sure you clicked **Train Face Data** *after* capturing — capturing
  alone only saves the image, training is what builds the embeddings
  RoboCamEngine compares against.
- Try capturing 3-5 samples per person from slightly different angles —
  this mirrors how the original Python recognizer benefits from more
  reference embeddings per person.
- The match threshold (`0.1` Euclidean distance) is copied directly from
  RoboCam's own code — it's intentionally strict, matching the original
  library's behavior.

### Performance is slow / laggy
The detector model runs on CPU in WASM by default (set in `main.jsx`).
This is the most broadly compatible option across Windows machines. If
your detection feels too slow, you can experiment with switching the
backend to `webgl` in `main.jsx` (`tf.setBackend('webgl')`), which uses
your GPU — faster on most machines, but slightly less consistent across
different graphics drivers.

### `npm install` fails or hangs
Run it while still on your **normal internet connection**, before
switching WiFi to the camera's isolated network.

---

## Captured face data and privacy

Captured face images are stored only in your browser's local IndexedDB
storage (`robocam-faces` database) — nothing is uploaded anywhere. This
data persists across page reloads but is specific to the browser profile
you're using; clearing browser site data will remove it, equivalent to
deleting RoboCam's `res/face/` folder in the Python version.
