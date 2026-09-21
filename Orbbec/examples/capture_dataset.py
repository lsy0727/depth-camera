#!/usr/bin/env python3
"""DCW2 RGB+Depth dataset capture tool (Windows only, camera plugged
directly into this PC).

Stills (RGB + Depth) come from dcw2_capture_daemon (built from
3rd_party/openni/src/dcw2_capture_daemon.cpp via
3rd_party/openni/src/build_windows.bat), which grabs a Color frame and a
Depth frame already aligned to the color frame's pixel grid from the Orbbec
SDK and streams them out as raw bytes. Both are resized to the requested
still resolution (RGB and Depth always match) and saved together with
per-frame metadata for later YOLO/COCO labeling.

The live preview only pulls RGB (via a lightweight daemon request that
skips the depth-alignment work and, when possible, forwards the color
sensor's native JPEG bytes instead of raw pixels). The full RGB+Depth pair
is only fetched once, at the moment you press s/SPACE to save a snapshot.

Video ('r' key) is recorded at the Color stream's native resolution (FHD)
via cv2.VideoWriter, independent of --res (which only affects saved
stills/preview).

Setup (one-time):
    1) Build the daemon: 3rd_party\\openni\\src\\build_windows.bat
       (needs the Windows Orbbec SDK + Visual Studio C++ build tools;
       see that file for details.)
    2) pip install opencv-python numpy
    3) Plug the camera into this PC.

Usage:
    python examples\\capture_dataset.py --class-name banner
    python examples\\capture_dataset.py --class-name banner --res 640x400

Keys (preview window):
    s / SPACE  save one RGB+Depth snapshot pair (still, resized per --res)
    a          toggle auto-save on/off (saves 6 RGB+Depth pairs/sec while on)
    r          start/stop video recording (native/FHD resolution, .mp4)
    q / ESC    quit

Headless mode (--headless, no display available) reads the same commands
as lines from the terminal instead ("s", "a", "r", "q" + Enter).

Output layout:
    <root>/<class_name>/
        image_<res>/<N>.jpg     RGB snapshot (640x480 or 640x400)
        depth_<res>/<N>.png     16-bit depth snapshot, same resolution as RGB
        meta_<res>/<N>.json     per-frame metadata
        video/<ts>.mp4          recorded video, native (FHD) resolution
        dataset_info.json       camera/session info for the labeling step
Existing files are never overwritten: numbering continues from the highest
index already present for the chosen class + resolution.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

if getattr(sys, "frozen", False):
    # Running as a PyInstaller-built .exe: lib/ (daemon + DLLs) is expected
    # to sit next to this .exe, not next to the (bundled, temp-extracted)
    # source -- see build_exe.bat.
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DAEMON = os.path.join(ROOT_DIR, "lib", "dcw2_capture_daemon.exe")
AUTO_SAVE_HZ = 6.0


class DaemonError(RuntimeError):
    pass


class CaptureDaemon:
    """Spawns dcw2_capture_daemon.exe locally and talks to it over stdin/stdout."""

    def __init__(self, daemon_path):
        if not os.path.isfile(daemon_path):
            raise DaemonError(
                f"daemon binary not found: {daemon_path}\n"
                f"Build it first: 3rd_party\\openni\\src\\build_windows.bat"
            )
        self.proc = subprocess.Popen(
            [daemon_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        self.rfile = self.proc.stdout
        self.wfile = self.proc.stdin

        ready = self.rfile.readline().decode("utf-8", "replace").strip()
        parts = ready.split("|")
        if not ready.startswith("READY") or len(parts) != 3:
            raise DaemonError(f"daemon did not report READY (got: {ready!r})")
        self.device_name = parts[1]
        self.device_serial = parts[2]

    def _read_exact(self, n):
        """self.rfile.read(n) is not guaranteed to return all n bytes in one
        call -- loop until we have everything, or the stream is exhausted."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self.rfile.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def get_frame(self):
        """Returns (rgb_uint8 HxWx3, depth_uint16_mm HxW) or None on failure."""
        self.wfile.write(b"GET\n")
        self.wfile.flush()
        header = self.rfile.readline().decode("ascii", "replace").strip()
        if not header.startswith("OK"):
            return None
        _, w_s, h_s = header.split()
        w, h = int(w_s), int(h_s)

        color_bytes = self._read_exact(w * h * 3)
        depth_bytes = self._read_exact(w * h * 2)
        if len(color_bytes) != w * h * 3 or len(depth_bytes) != w * h * 2:
            return None

        rgb = np.frombuffer(color_bytes, dtype=np.uint8).reshape(h, w, 3)
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w)
        return rgb, depth

    def get_color_frame(self):
        """Lightweight color-only request for a smooth live preview: skips the
        depth alignment work and, when possible, returns the color sensor's
        native JPEG bytes instead of raw pixels.
        Returns a BGR uint8 HxWx3 array (ready for cv2.imshow) or None."""
        self.wfile.write(b"GETC\n")
        self.wfile.flush()
        header = self.rfile.readline().decode("ascii", "replace").strip()
        if not header.startswith("OKC"):
            return None
        _, w_s, h_s, encoding, n_s = header.split()
        w, h, n = int(w_s), int(h_s), int(n_s)

        payload = self._read_exact(n)
        if len(payload) != n:
            return None

        if encoding == "MJPG":
            buf = np.frombuffer(payload, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return bgr
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        try:
            self.wfile.write(b"QUIT\n")
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Recorder:
    """Records the live Color frames (native/FHD resolution) via
    cv2.VideoWriter, independent of the --res used for stills/preview."""

    def __init__(self, video_dir, fps):
        self.video_dir = video_dir
        self.fps = fps
        self.writer = None
        self.path = None
        # start()/stop() run on the main thread (in response to 'r'), while
        # write_frame() is called from run_headless's background feed
        # thread -- without this, stop() releasing the writer can race a
        # concurrent write_frame() still using it (seen in practice: cv2
        # "Failed to write frame" / invalid pts right at a stop/start).
        self._lock = threading.Lock()

    @property
    def recording(self):
        return self.writer is not None

    def start(self, sample_frame_bgr):
        with self._lock:
            if self.writer is not None or sample_frame_bgr is None:
                return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.video_dir, f"{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = sample_frame_bgr.shape[:2]
            self.writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
            self.path = path
            return path

    def write_frame(self, bgr):
        with self._lock:
            if self.writer is not None:
                self.writer.write(bgr)

    def stop(self):
        with self._lock:
            if self.writer is None:
                return None
            path = self.path
            self.writer.release()
            self.writer = None
            self.path = None
            return path

    def close(self):
        self.stop()


def next_index(image_dir):
    max_idx = 0
    for path in glob.glob(os.path.join(image_dir, "*.jpg")):
        m = re.match(r"^(\d+)\.jpg$", os.path.basename(path))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


class Session:
    def __init__(self, root, class_name, res_w, res_h, camera_name, camera_serial, fps):
        self.res_w, self.res_h = res_w, res_h
        self.res_tag = f"{res_w}x{res_h}"
        self.fps = fps
        self.class_name = class_name

        self.class_dir = os.path.join(root, class_name)
        self.image_dir = os.path.join(self.class_dir, f"image_{self.res_tag}")
        self.depth_dir = os.path.join(self.class_dir, f"depth_{self.res_tag}")
        self.meta_dir = os.path.join(self.class_dir, f"meta_{self.res_tag}")
        self.video_dir = os.path.join(self.class_dir, "video")  # native resolution, independent of --res
        for d in (self.image_dir, self.depth_dir, self.meta_dir, self.video_dir):
            os.makedirs(d, exist_ok=True)

        self.next_idx = next_index(self.image_dir)
        self.camera_name = camera_name
        self.camera_serial = camera_serial
        self._write_dataset_info()

    def _write_dataset_info(self):
        info_path = os.path.join(self.class_dir, "dataset_info.json")
        info = {
            "class_name": self.class_name,
            "resolution": self.res_tag,
            "fps": self.fps,
            "camera": {
                "model": "DaBai DCW2",
                "raw_name": self.camera_name,
                "mxid": self.camera_serial,
            },
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def fit(self, rgb, depth):
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if (rgb_bgr.shape[1], rgb_bgr.shape[0]) != (self.res_w, self.res_h):
            rgb_bgr = cv2.resize(rgb_bgr, (self.res_w, self.res_h), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (self.res_w, self.res_h), interpolation=cv2.INTER_NEAREST)
        return rgb_bgr, depth

    def fit_color(self, bgr):
        """Same resize as fit(), color-only -- for the live preview loop,
        which no longer pulls depth every frame (see get_color_frame)."""
        if (bgr.shape[1], bgr.shape[0]) != (self.res_w, self.res_h):
            bgr = cv2.resize(bgr, (self.res_w, self.res_h), interpolation=cv2.INTER_AREA)
        return bgr

    def save_snapshot(self, rgb_bgr, depth):
        idx = self.next_idx
        self.next_idx += 1

        rgb_path = os.path.join(self.image_dir, f"{idx}.jpg")
        depth_path = os.path.join(self.depth_dir, f"{idx}.png")
        meta_path = os.path.join(self.meta_dir, f"{idx}.json")

        cv2.imwrite(rgb_path, rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(depth_path, depth)

        meta = {
            "frame_id": idx,
            "session_folder": self.class_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "fps": self.fps,
            "rgb_path": f"image_{self.res_tag}/{idx}.jpg",
            "depth_path": f"depth_{self.res_tag}/{idx}.png",
            "camera": {
                "model": "DaBai DCW2",
                "raw_name": self.camera_name,
                "mxid": self.camera_serial,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return idx


def run_preview(daemon, session, recorder, fps):
    interval = 1.0 / fps
    auto_save_interval = 1.0 / AUTO_SAVE_HZ
    win = "DCW2 capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, session.res_w, session.res_h)
    print(f"[안내] s/SPACE: 스냅샷 | a: 자동저장(초당 {AUTO_SAVE_HZ:g}장) on/off | r: 녹화 시작/종료 | q: 종료")

    auto_save = False
    last_auto_save = 0.0
    last = 0.0
    while True:
        now = time.time()
        if now - last < interval:
            time.sleep(max(0.0, interval - (now - last)))
        last = time.time()

        # Live preview only pulls RGB (fast, often JPEG); the depth-aligned
        # pair is only fetched (via get_frame) on demand -- either a manual
        # 's' press, or here, on auto_save's own timer.
        bgr = daemon.get_color_frame()
        if bgr is None:
            continue
        # Recording gets the frame at its native (FHD) resolution, not
        # resized/overlaid -- only the on-screen preview and saved stills
        # are downscaled to --res.
        recorder.write_frame(bgr)

        if auto_save and last - last_auto_save >= auto_save_interval:
            last_auto_save = last
            pair = daemon.get_frame()
            if pair is not None:
                rgb, depth = pair
                rgb_bgr, depth = session.fit(rgb, depth)
                idx = session.save_snapshot(rgb_bgr, depth)
                print(f"[자동저장] #{idx} -> {session.image_dir}, {session.depth_dir}")

        view = session.fit_color(bgr)
        cv2.putText(view, f"class={session.class_name} saved={session.next_idx - 1}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if recorder.recording:
            cv2.circle(view, (20, 50), 8, (0, 0, 255), -1)
            cv2.putText(view, "REC", (35, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if auto_save:
            cv2.circle(view, (20, 80), 8, (0, 255, 0), -1)
            cv2.putText(view, "AUTO", (35, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord(' ')):
            pair = daemon.get_frame()
            if pair is None:
                print("[실패] 프레임을 받지 못했습니다.")
            else:
                rgb, depth = pair
                rgb_bgr, depth = session.fit(rgb, depth)
                idx = session.save_snapshot(rgb_bgr, depth)
                print(f"[저장] #{idx} -> {session.image_dir}, {session.depth_dir}")
        elif key == ord('a'):
            auto_save = not auto_save
            last_auto_save = 0.0
            print(f"[자동저장 {'시작' if auto_save else '종료'}]")
        elif key == ord('r'):
            if recorder.recording:
                print(f"[녹화 종료] {recorder.stop()}")
            else:
                print(f"[녹화 시작] {recorder.start(bgr)}")
        elif key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


def run_headless(daemon, session, recorder, fps):
    # Recorder and auto-save both need a steady stream of frames pulled from
    # the same daemon connection the 's'/on-demand snapshot uses -- one lock
    # keeps those calls from interleaving (the protocol isn't safe to call
    # from two threads at once).
    lock = threading.Lock()
    stop_event = threading.Event()
    auto_save_event = threading.Event()

    def feed_loop():
        # Only pull frames while actually recording -- otherwise this loop
        # would keep hammering the daemon for GETC frames indefinitely,
        # competing with the occasional on-demand GET (snapshot) call and,
        # in practice, making the depth frame that GET returns right after
        # a recording stops come back empty.
        interval = 1.0 / fps
        last = 0.0
        while not stop_event.is_set():
            if not recorder.recording:
                time.sleep(0.05)
                continue
            now = time.time()
            if now - last < interval:
                time.sleep(max(0.0, interval - (now - last)))
            last = time.time()
            with lock:
                bgr = daemon.get_color_frame()
            if bgr is not None:
                recorder.write_frame(bgr)  # native (FHD) resolution, not resized to --res

    def auto_save_loop():
        interval = 1.0 / AUTO_SAVE_HZ
        last = 0.0
        while not stop_event.is_set():
            if not auto_save_event.is_set():
                time.sleep(0.05)
                continue
            now = time.time()
            if now - last < interval:
                time.sleep(max(0.0, interval - (now - last)))
            last = time.time()
            with lock:
                pair = daemon.get_frame()
            if pair is not None:
                rgb, depth = pair
                rgb_bgr, depth = session.fit(rgb, depth)
                idx = session.save_snapshot(rgb_bgr, depth)
                print(f"[자동저장] #{idx} -> {session.image_dir}, {session.depth_dir}")

    feed_thread = threading.Thread(target=feed_loop, daemon=True)
    feed_thread.start()
    auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
    auto_save_thread.start()

    print(f"[안내] 명령 입력: s(스냅샷) / a(자동저장 초당 {AUTO_SAVE_HZ:g}장 on/off) / r(녹화 시작/종료) / q(종료)")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("s", ""):
            with lock:
                pair = daemon.get_frame()
            if pair is None:
                print("[실패] 프레임을 받지 못했습니다.")
                continue
            rgb, depth = pair
            rgb_bgr, depth = session.fit(rgb, depth)
            idx = session.save_snapshot(rgb_bgr, depth)
            print(f"[저장] #{idx} -> {session.image_dir}, {session.depth_dir}")
        elif cmd == "a":
            if auto_save_event.is_set():
                auto_save_event.clear()
                print("[자동저장 종료]")
            else:
                auto_save_event.set()
                print("[자동저장 시작]")
        elif cmd == "r":
            if recorder.recording:
                print(f"[녹화 종료] {recorder.stop()}")
            else:
                with lock:
                    sample = daemon.get_color_frame()
                if sample is None:
                    print("[실패] 프레임을 받지 못해 녹화를 시작할 수 없습니다.")
                else:
                    print(f"[녹화 시작] {recorder.start(sample)}")
        elif cmd == "q":
            break

    stop_event.set()
    feed_thread.join(timeout=2)
    auto_save_thread.join(timeout=2)


def _prompt_class_name():
    while True:
        name = input("Class 이름 입력 (예: banner): ").strip()
        if name:
            return name
        print("[오류] class 이름을 입력해주세요.")


def _prompt_res():
    print("해상도 선택 (동영상은 이 설정과 무관하게 항상 FHD로 저장됩니다)")
    print("  1) 640x480 (기본값)")
    print("  2) 640x400")
    choice = input("선택 [1/2] (Enter=1): ").strip()
    return "640x400" if choice == "2" else "640x480"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", "-c", default=None, help="dataset class/object name (output subfolder); "
                                                                   "prompted interactively if omitted")
    parser.add_argument("--root", default=os.path.join(ROOT_DIR, "captures"), help="dataset output root")
    parser.add_argument("--res", choices=["640x480", "640x400"], default=None,
                         help="saved still-image resolution (video is always FHD regardless of this); "
                              "prompted interactively if --class-name is omitted, else defaults to 640x480")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="preview/recording loop rate cap (just a ceiling -- actual rate is "
                              "whatever the camera/decode can sustain); also recorded into metadata")
    parser.add_argument("--daemon", default=DEFAULT_DAEMON, help="path to dcw2_capture_daemon.exe")
    parser.add_argument("--headless", action="store_true", help="no display available; drive via typed commands")
    args = parser.parse_args()

    # No --class-name means this was launched by double-clicking the .exe
    # (no command line), not from a script/terminal that already knows what
    # it wants -- ask interactively instead of erroring out on a missing
    # required argument, and pause before exit so the console window
    # doesn't just flash closed.
    interactive = args.class_name is None
    if interactive:
        print("=== DCW2 RGB+Depth 데이터셋 캡처 ===")
        args.class_name = _prompt_class_name()
        args.res = _prompt_res()
    elif args.res is None:
        args.res = "640x480"

    try:
        res_w, res_h = (int(v) for v in args.res.split("x"))

        daemon = CaptureDaemon(args.daemon)
        print(f"[연결] {daemon.device_name} (S/N {daemon.device_serial})")

        session = Session(args.root, args.class_name, res_w, res_h,
                           daemon.device_name, daemon.device_serial, args.fps)
        print(f"[세션] class={args.class_name} res={args.res} 다음 인덱스={session.next_idx}")

        recorder = Recorder(session.video_dir, args.fps)

        try:
            if args.headless:
                run_headless(daemon, session, recorder, args.fps)
            else:
                try:
                    run_preview(daemon, session, recorder, args.fps)
                except cv2.error as e:
                    print(f"[경고] 화면 출력을 사용할 수 없습니다 ({e}); --headless 모드로 전환합니다.")
                    run_headless(daemon, session, recorder, args.fps)
        finally:
            recorder.close()
            daemon.close()
    except Exception as e:
        if interactive:
            print(f"[오류] {e}")
            input("Enter 키를 눌러 종료...")
            sys.exit(1)
        raise
    else:
        if interactive:
            input("종료하려면 Enter 키를 누르세요...")


if __name__ == "__main__":
    main()
