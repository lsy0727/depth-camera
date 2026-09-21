# DCW2 RGB+Depth Dataset Capture

Windows tool for capturing RGB+Depth still pairs and FHD video from an Orbbec
DaBai DCW2 camera, for building YOLO/COCO-style datasets.

## Quick start (another PC, no Python needed)

1. Install the camera driver: [`Driver/SensorDriver_V4.3.0.22.exe`](Driver/SensorDriver_V4.3.0.22.exe)
2. Copy these two items to the target PC, keeping them next to each other:
   ```
   capture_dataset.exe
   lib/                 (daemon + Orbbec SDK DLLs)
   ```
3. Plug in the DaBai DCW2 camera, then double-click `capture_dataset.exe`.
4. Follow the prompts: class name, then resolution (`1` = 640x480, `2` = 640x400).

No Python install is required on the target PC — everything is bundled into
the `.exe`.

## Running from source (this PC)

```bash
pip install opencv-python numpy
python examples/capture_dataset.py --class-name banner
```

or double-click [`capture_dataset.bat`](capture_dataset.bat).

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--class-name`, `-c` | *(prompted if omitted)* | dataset class/object name (output subfolder) |
| `--res` | `640x480` | saved still-image resolution (`640x480` or `640x400`); video is always FHD regardless |
| `--root` | `captures/` | dataset output root |
| `--fps` | `30.0` | preview/recording loop rate cap (a ceiling, not a guarantee) |
| `--headless` | off | no display available; drive via typed commands instead of the preview window |
| `--daemon` | `lib/dcw2_capture_daemon.exe` | path to the capture daemon binary |

Running with no `--class-name` (e.g. double-clicking the `.exe`) drops into
interactive prompts for class name and resolution instead of erroring out.

### Keys (preview window) / commands (`--headless`)

| Key | Command | Action |
|---|---|---|
| `s` / SPACE | `s` | save one RGB+Depth snapshot pair (resized per `--res`) |
| `a` | `a` | toggle auto-save on/off — saves ~6 RGB+Depth pairs/sec while on |
| `r` | `r` | start/stop video recording (native FHD, `.mp4`) |
| `q` / ESC | `q` | quit |

## Output layout

```
captures/<class_name>/
    image_<res>/<N>.jpg     RGB snapshot
    depth_<res>/<N>.png     16-bit depth snapshot, same resolution as RGB
    meta_<res>/<N>.json     per-frame metadata
    video/<ts>.mp4          recorded video, native (FHD) resolution
    dataset_info.json       camera/session info for the labeling step
```

Existing files are never overwritten — numbering continues from the highest
index already present for the chosen class + resolution.

## Rebuilding the .exe

```bash
build_exe.bat
```

This creates its own throwaway virtual environment (`.build_venv`) with the
standard PyPI `opencv-python` wheel and builds from there — **not** whatever
`cv2` this machine's global Python has installed. This matters: if the
global Python has a custom/locally-built OpenCV (e.g. built from source for
CUDA), building with it produces an `.exe` that crashes on every other PC
with:

```
ImportError: ERROR: recursion is detected during loading of "cv2" binary
extensions. Check OpenCV installation.
```

A custom build's `cv2/config.py` / `cv2/config-3.py` bundle this machine's
own absolute file paths, which don't exist elsewhere and confuse OpenCV's
loader once frozen. The standard wheel has no such hardcoded paths, so
always build through `build_exe.bat`'s venv rather than the system Python.

Output: `capture_dataset.exe`, written to the repo root next to `lib/`.

## Repository layout

```
capture_dataset.exe     standalone build (see Quick start)
capture_dataset.bat     launcher for running from source via Python
build_exe.bat           rebuilds capture_dataset.exe
examples/
    capture_dataset.py  the capture tool itself
Driver/                 Orbbec camera driver installer
lib/                    daemon binary + Orbbec SDK DLLs the .exe/script need at runtime
OrbbecSDK_v1.10.35/     vendored Orbbec SDK (headers, docs, examples)
```

See the docstring at the top of [`examples/capture_dataset.py`](examples/capture_dataset.py)
for implementation details (daemon protocol, preview vs. save frame paths,
etc).
