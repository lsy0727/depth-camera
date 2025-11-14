#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import depthai as dai
import cv2
import numpy as np
import os
import json
import time
import re
from datetime import datetime
import argparse

# =========================================================
# 유틸: 숫자 폴더 / 프레임 인덱스 / 사용 폴더(JSON) 관리
# =========================================================

def list_numeric_dirs(root: str) -> set[int]:
    os.makedirs(root, exist_ok=True)
    pat = re.compile(r"^(\d+)$")
    nums = set()
    for name in os.listdir(root):
        m = pat.match(name)
        if m:
            nums.add(int(m.group(1)))
    return nums

def load_global_index(root: str) -> int:
    idx_path = os.path.join(root, "frame_index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("last_index", 0))
        except Exception:
            return 0
    return 0

def update_global_index(root: str, new_index: int) -> None:
    idx_path = os.path.join(root, "frame_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"last_index": int(new_index)}, f, indent=2, ensure_ascii=False)

# ---- 저장된 폴더 번호(JSON) ----
def _dir_usage_path(root: str) -> str:
    return os.path.join(root, "dir_usage.json")

def load_saved_dirs(root: str) -> set[int]:
    """
    dataset/dir_usage.json에서 이미 '프레임이 저장된' 폴더 번호들을 복원.
    파일이 없거나 파싱 실패 시 빈 집합 반환.
    구조: {"saved_dirs": [1,3,5], "updated_at": "..."}
    """
    path = _dir_usage_path(root)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = data.get("saved_dirs", [])
        saved = set(int(x) for x in arr if int(x) >= 1)
        return saved
    except Exception:
        return set()

def save_saved_dirs(root: str, saved_dirs: set[int]) -> None:
    """
    saved_dirs를 유니크/소트하여 dataset/dir_usage.json에 저장(중복 방지).
    """
    path = _dir_usage_path(root)
    payload = {
        "saved_dirs": sorted(int(x) for x in saved_dirs if int(x) >= 1),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def next_missing_from_saved(saved_dirs: set[int]) -> int:
    """
    JSON(saved_dirs)에 없는 '가장 작은 양의 정수'를 반환.
    예) {}->1, {5}->{1}, {1,2,5}->{3}, {1,2,3}->{4}
    """
    n = 1
    while n in saved_dirs:
        n += 1
    return n

# =========================================================
# 안전 저장 래퍼 (경로 유니코드/긴 경로 대응)
# =========================================================

def _ensure_contiguous(arr: np.ndarray, dtype=None) -> np.ndarray:
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr

def _write_png_via_imencode(path: str, array: np.ndarray, *, compression: int = 0) -> bool:
    """
    경로 이슈 회피: OpenCV가 파일을 직접 쓰지 않게 하고,
    메모리 인코딩한 버퍼를 파이썬 바이너리 파일로 저장.
    """
    try:
        # PNG 압축 레벨 지정 (0=무압축, 9=최대압축)
        ok, buf = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if not ok:
            print(f"[WARN] imencode('.png') failed for: {path}")
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"[WARN] write via imencode failed: {e}")
        return False

def _imwrite_png_safe(path: str, array: np.ndarray) -> bool:
    """
    1) imencode + python open 으로 저장 시도 (유니코드/긴 경로 OK)
    2) 실패 시 Pillow 폴백
    3) 그래도 실패하면 마지막으로 cv2.imwrite 시도
    """
    # 1) imencode 경로
    if _write_png_via_imencode(path, array, compression=0):
        return True

    # 2) Pillow 폴백
    try:
        from PIL import Image
        # dtype/채널 보고 모드 선택
        if array.dtype == np.uint16:
            img = Image.fromarray(array, mode="I;16")
        else:
            # 채널 수 판단
            if array.ndim == 2:
                img = Image.fromarray(array, mode="L")
            else:
                # OpenCV는 RGB로 세팅되어 있음
                img = Image.fromarray(array, mode="RGB")
        img.save(path, format="PNG", compress_level=0)
        print(f"[INFO] Saved via Pillow fallback: {path}")
        return True
    except Exception as e:
        print(f"[WARN] Pillow fallback failed: {e}")

    # 3) 최후: cv2.imwrite (성공하면 좋고, 실패면 False 반환)
    try:
        ok = cv2.imwrite(path, array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            print(f"[WARN] cv2.imwrite returned False: {path}")
        return ok
    except Exception as e:
        print(f"[ERROR] cv2.imwrite raised: {e}")
        return False

# =========================================================
# 수집기
# =========================================================

class OAKDDataCollector:
    def __init__(self, output_root: str = "dataset"):
        # 루트 (절대경로화는 유지하되, 저장 래퍼로 경로 문제 해결)
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)

        # 현재(선택된) 세션 폴더 상태
        self.session_dir = None
        self.session_id = None
        self.rgb_dir = None
        self.depth_dir = None
        self.ann_dir = None

        # 프레임 인덱스(전역)
        self.global_last_index = load_global_index(self.output_root)
        self.next_idx = self.global_last_index + 1

        # 표시/상태
        self.mouse_pos = (0, 0)
        self.current_depth_frame = None
        self.current_fps = 0.0
        self._fps_counter = 0
        self._fps_t0 = time.time()

        # 카메라 정보
        self.camera_model = "unknown"
        self.camera_model_raw = "unknown"
        self.camera_mxid = None

        # ----- JSON 기반 '저장된 폴더' 세트 -----
        self.saved_dirs = load_saved_dirs(self.output_root)  # set[int]
        # 프로그램 시작 시 기본 저장 디렉토리 = JSON에 없는 가장 작은 번호
        self.dir_base = next_missing_from_saved(self.saved_dirs)
        # 저장 직전에 일시 조정하는 오프셋(이번 실행에서만 유지)
        self.save_dir_offset = 0

        self.setup_pipeline()

    # ---------- 카메라 모델 감지 ----------
    @staticmethod
    def _normalize_camera_model(raw: str | None) -> str:
        if not raw:
            return "unknown"
        s = str(raw).lower()
        if "lite" in s:
            return "oak-d lite"
        if "s2" in s:
            return "oak-d s2"
        if "wide" in s:
            return "oak-d wide"
        return "unknown"

    def _detect_camera_model(self, device: dai.Device) -> None:
        raw = None
        for attr in ("getDeviceName", "getProductName", "getBoardName"):
            func = getattr(device, attr, None)
            if callable(func):
                try:
                    raw = func()
                    if raw:
                        break
                except Exception:
                    pass
        if not raw:
            try:
                calib = device.readCalibration()
                eeprom = calib.getEepromData()
                raw = getattr(eeprom, "productName", None) or getattr(eeprom, "boardName", None)
            except Exception:
                pass
        try:
            self.camera_mxid = device.getMxId()
        except Exception:
            self.camera_mxid = None
        self.camera_model_raw = str(raw) if raw else "unknown"
        self.camera_model = self._normalize_camera_model(self.camera_model_raw)

    # ---------- 현재 저장 대상 디렉토리 ----------
    def _current_target_session_number(self) -> int:
        """
        Dir(기본) + 일시 오프셋.
        하한: 1, 상한: 제한 없음
        """
        target = self.dir_base + self.save_dir_offset
        if target < 1:
            target = 1
        return target

    def _ensure_session_created(self, target_session_number: int | None = None):
        if target_session_number is None:
            target_session_number = self._current_target_session_number()
        if self.session_dir is not None and self.session_id == str(target_session_number):
            return

        session_dir = os.path.join(self.output_root, str(target_session_number))
        rgb_dir = os.path.join(session_dir, "rgb")
        depth_dir = os.path.join(session_dir, "depth")
        ann_dir = os.path.join(session_dir, "annotations")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        self.session_dir = session_dir
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.ann_dir = ann_dir
        self.session_id = str(target_session_number)

        print(f"[세션 설정] 저장 대상 폴더 → {self.session_dir}")

    # ---------- 파이프라인 ----------
    def setup_pipeline(self) -> None:
        self.pipeline = dai.Pipeline()

        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(640, 400)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.cam_rgb.setFps(30)

        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setConfidenceThreshold(255)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.initialConfig.setDisparityShift(0)
        self.stereo.setOutputSize(640, 400)

        cfg = self.stereo.initialConfig.get()
        cfg.postProcessing.spatialFilter.enable = False
        self.stereo.initialConfig.set(cfg)

        self.mono_left = self.pipeline.create(dai.node.MonoCamera)
        self.mono_right = self.pipeline.create(dai.node.MonoCamera)
        self.mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        self.mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        self.mono_left.setFps(30)
        self.mono_right.setFps(30)

        self.mono_left.out.link(self.stereo.left)
        self.mono_right.out.link(self.stereo.right)

        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)

        self.xout_depth = self.pipeline.create(dai.node.XLinkOut)
        self.xout_depth.setStreamName("depth")
        self.stereo.depth.link(self.xout_depth.input)

    # ---------- 표시 유틸 ----------
    @staticmethod
    def _draw_text(img, text, org=(10, 30), scale=0.7):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _update_fps(self):
        self._fps_counter += 1
        if self._fps_counter >= 30:
            t1 = time.time()
            self.current_fps = self._fps_counter / (t1 - self._fps_t0)
            self._fps_counter = 0
            self._fps_t0 = t1

    # ---------- 마우스 콜백 ----------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

    # ---------- 저장 ----------
    def save_frame_data(self, frame_rgb, frame_depth, fps: float | None = None) -> int:
        # 저장 대상 디렉토리 보장/전환
        target_n = self._current_target_session_number()
        self._ensure_session_created(target_session_number=target_n)

        idx = self.next_idx
        base = f"frame{idx}"  # ← 파일명 접두사

        # RGB 저장 (640x400, uint8, 경로 안전 저장)
        rgb_out = cv2.resize(frame_rgb, (640, 400), interpolation=cv2.INTER_LINEAR)
        rgb_out = _ensure_contiguous(rgb_out, dtype=np.uint8)
        rgb_path = os.path.join(self.rgb_dir, f"{base}.png")
        ok_rgb = _imwrite_png_safe(rgb_path, rgb_out)
        if not ok_rgb:
            print(f"[ERROR] RGB 저장 실패: {rgb_path}")

        # Depth 저장 (16-bit PNG, 경로 안전 저장)
        min_d, max_d = 100, 5000
        depth_clipped = np.clip(frame_depth, min_d, max_d) * (frame_depth > 0)
        depth_out = cv2.resize(depth_clipped, (640, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        depth_out = _ensure_contiguous(depth_out, dtype=np.uint16)
        depth_path = os.path.join(self.depth_dir, f"{base}.png")
        ok_depth = _imwrite_png_safe(depth_path, depth_out)
        if not ok_depth:
            print(f"[ERROR] Depth 저장 실패: {depth_path}")

        # 메타 저장 (파일명도 frame{idx}.json으로)
        meta = {
            "frame_id": base,
            "session_folder": self.session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "fps": float(fps or 0.0),
            "rgb_path": f"rgb/{base}.png",
            "depth_path": f"depth/{base}.png",
            "camera": {
                "model": self.camera_model,
                "raw_name": self.camera_model_raw,
                "mxid": self.camera_mxid
            }
        }
        with open(os.path.join(self.ann_dir, f"{base}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 프레임 인덱스 증가(디렉토리와 독립)
        self.next_idx += 1
        update_global_index(self.output_root, self.next_idx - 1)

        # ★ JSON 갱신: 실제 저장된 폴더 번호를 saved_dirs에 추가(중복 없이)
        saved_before = len(self.saved_dirs)
        self.saved_dirs.add(int(self.session_id))
        if len(self.saved_dirs) != saved_before:
            save_saved_dirs(self.output_root, self.saved_dirs)

        return idx  # 호출부 출력은 "frame{idx}"로 표시

    # ---------- 메인 루프 ----------
    def collect_data(self, max_frames: int | None = None, save_interval: int = 5) -> None:
        print("세션 폴더: (저장 시 생성/선택)")
        print("키보드: SPACE=저장, a=연속저장, 1=폴더 내리기(최소 1), 2=폴더 올리기(상한 없음), q/ESC=종료")

        continuous_save = False
        processed_frames = 0
        KEY_DOWN_NUM = ord('1')  # 디렉토리 낮춤
        KEY_UP_NUM   = ord('2')  # 디렉토리 올림

        try:
            with dai.Device(self.pipeline) as device:
                self._detect_camera_model(device)
                print(f"Camera detected → model: {self.camera_model}, raw: '{self.camera_model_raw}', mxid: {self.camera_mxid}")

                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=True)

                cv2.namedWindow("OAK-D RGB Camera", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow("OAK-D Depth Camera", cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback("OAK-D Depth Camera", self.mouse_callback)

                while True:
                    in_rgb = q_rgb.get()
                    in_depth = q_depth.get()
                    frame_rgb = in_rgb.getCvFrame()
                    frame_depth = in_depth.getFrame()
                    self.current_depth_frame = frame_depth
                    processed_frames += 1

                    # 화면 표시용
                    disp_rgb = cv2.resize(frame_rgb, (640, 400), interpolation=cv2.INTER_LINEAR)
                    min_d, max_d = 100, 5000
                    depth_clipped = np.clip(frame_depth, min_d, max_d) * (frame_depth > 0)
                    depth_norm = ((depth_clipped - min_d) / (max_d - min_d) * 255).astype(np.uint8)
                    disp_depth = cv2.cvtColor(
                        cv2.resize(depth_norm, (640, 400), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR
                    )

                    self._update_fps()

                    # 오버레이 (Dir/SaveDir/NextIdx)
                    target_n = self._current_target_session_number()
                    overlay = f"FPS: {self.current_fps:.1f} | Dir: {self.dir_base} | SaveDir: {target_n} | NextIdx: {self.next_idx}"
                    self._draw_text(disp_rgb, overlay, (10, 26), 0.65)
                    self._draw_text(disp_depth, overlay, (10, 26), 0.65)
                    if continuous_save:
                        self._draw_text(disp_rgb, "[AUTO SAVE]", (10, 56), 0.6)
                        self._draw_text(disp_depth, "[AUTO SAVE]", (10, 56), 0.6)

                    # 마우스 위치 depth 값 표시
                    mx, my = self.mouse_pos
                    if self.current_depth_frame is not None and 0 <= mx < 640 and 0 <= my < 400:
                        H, W = frame_depth.shape
                        sx, sy = W / 640.0, H / 400.0
                        ox, oy = int(mx * sx), int(my * sy)
                        if 0 <= ox < W and 0 <= oy < H:
                            dval = int(frame_depth[oy, ox])
                            if dval > 0:
                                cv2.drawMarker(disp_depth, (mx, my), (255, 255, 255),
                                               cv2.MARKER_CROSS, 10, 2)
                                txt = f"{dval} mm ({dval/10:.1f} cm)"
                                cv2.putText(disp_depth, txt, (mx + 10, my - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                                cv2.putText(disp_depth, txt, (mx + 10, my - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow("OAK-D RGB Camera", disp_rgb)
                    cv2.imshow("OAK-D Depth Camera", disp_depth)

                    # 키 입력
                    key = cv2.waitKey(1) & 0xFF
                    should_save = False
                    if key == ord(' '):
                        should_save = True
                    elif key == ord('a'):
                        continuous_save = not continuous_save
                        print(f"연속 저장: {'ON' if continuous_save else 'OFF'}")
                    elif key == ord('q') or key == 27:
                        break
                    elif key == KEY_UP_NUM:
                        # 상한 없음
                        self.save_dir_offset += 1
                        print(f"폴더 올리기 → SaveDir {self._current_target_session_number()}")
                    elif key == KEY_DOWN_NUM:
                        # 최소 1
                        if self._current_target_session_number() > 1:
                            self.save_dir_offset -= 1
                            if self._current_target_session_number() < 1:
                                self.save_dir_offset += 1  # 보정
                            else:
                                print(f"폴더 내리기 → SaveDir {self._current_target_session_number()}")

                    # 자동 저장
                    if continuous_save and (processed_frames % max(1, save_interval) == 0):
                        should_save = True

                    if should_save:
                        idx = self.save_frame_data(frame_rgb, frame_depth, self.current_fps)
                        print(f"저장됨 → 폴더 {self.session_id}, 프레임 {idx} (FPS {self.current_fps:.1f})")
                        # 최대 프레임 제한
                        if max_frames is not None and (idx - self.global_last_index) >= max_frames:
                            print(f"최대 프레임 수({max_frames}) 도달. 종료합니다.")
                            break

        except Exception as e:
            print(f"[오류] {e}")
            print("OAK-D 연결/권한/케이블/전원 상태를 확인하세요.")
        finally:
            cv2.destroyAllWindows()
            if self.session_dir is not None:
                print(f"데이터 수집 완료. 마지막 프레임 번호: {self.next_idx-1}")
                print(f"세션 경로: {self.session_dir}")
            else:
                print("데이터 수집 종료(이번 실행에서는 저장하지 않아 세션 폴더가 생성되지 않았습니다).")

# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="OAK-D 카메라 데이터 수집기 (카톡 경로 대응: 안전 저장)")
    parser.add_argument("--output", "-o", default="dataset", help="루트 디렉토리 (기본: dataset)")
    parser.add_argument("--max-frames", "-m", type=int, help="최대 저장 프레임 수")
    parser.add_argument("--interval", "-i", type=int, default=5, help="연속 저장 간격(프레임 단위)")
    args = parser.parse_args()

    collector = OAKDDataCollector(output_root=args.output)
    collector.collect_data(max_frames=args.max_frames, save_interval=args.interval)

if __name__ == "__main__":
    main()
