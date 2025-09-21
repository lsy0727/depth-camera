import depthai as dai
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import argparse

class OAKDDataCollector:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        self.processed_count = 0
        self.saved_count = 0
        self.current_depth_frame = None
        self.mouse_pos = (0, 0)
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """OAK-D 파이프라인 설정 (FPS 및 정확도 균형)"""
        self.pipeline = dai.Pipeline()
        
        # 컬러 카메라 설정
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(640, 400)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.cam_rgb.setFps(30)
        
        # 스테레오 깊이 카메라 설정
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setConfidenceThreshold(255)  # 245 -> 255
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.initialConfig.setDisparityShift(0)
        self.stereo.setOutputSize(640, 400)
        
        # 공간 필터 비활성화
        cfg = self.stereo.initialConfig.get()
        cfg.postProcessing.spatialFilter.enable = False
        self.stereo.initialConfig.set(cfg)
        
        # 모노 카메라 설정
        self.mono_left = self.pipeline.create(dai.node.MonoCamera)
        self.mono_right = self.pipeline.create(dai.node.MonoCamera)
        self.mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        self.mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        self.mono_left.setFps(30)
        self.mono_right.setFps(30)
        
        # 연결
        self.mono_left.out.link(self.stereo.left)
        self.mono_right.out.link(self.stereo.right)
        
        # 출력 XLink 설정
        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        
        self.xout_depth = self.pipeline.create(dai.node.XLinkOut)
        self.xout_depth.setStreamName("depth")
        self.stereo.depth.link(self.xout_depth.input)
    
    def save_frame_data(self, frame_rgb, frame_depth, fps=None):
        """프레임 데이터 저장 (Depth 클리핑 적용)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_id = f"frame{self.saved_count}"
        
        # RGB 프레임 저장
        frame_rgb_saved = cv2.resize(frame_rgb, (640, 400))
        rgb_path = os.path.join(self.rgb_dir, f"{frame_id}.png")
        cv2.imwrite(rgb_path, frame_rgb_saved, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Depth 프레임 클리핑 및 저장
        min_depth = 100
        max_depth = 5000
        frame_depth_clipped = np.clip(frame_depth, min_depth, max_depth) * (frame_depth > 0)
        frame_depth_saved = cv2.resize(frame_depth_clipped, (640, 400), interpolation=cv2.INTER_NEAREST)
        depth_path = os.path.join(self.depth_dir, f"{frame_id}.png")
        cv2.imwrite(depth_path, frame_depth_saved.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        valid_depth = frame_depth[frame_depth > 0]
        depth_stats = {}
        if len(valid_depth) > 0:
            depth_stats = {
                "valid_pixels": int(len(valid_depth)),
                "total_pixels": int(frame_depth.size),
                "valid_ratio": float(len(valid_depth) / frame_depth.size),
                "min_depth": float(valid_depth.min()),
                "max_depth": float(valid_depth.max()),
                "mean_depth": float(valid_depth.mean()),
                "std_depth": float(valid_depth.std()),
                "noise_ratio": float(np.sum(valid_depth > 5000) / len(valid_depth)) if len(valid_depth) > 0 else 0.0
            }
        else:
            depth_stats = {
                "valid_pixels": 0,
                "total_pixels": int(frame_depth.size),
                "valid_ratio": 0.0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "mean_depth": 0.0,
                "std_depth": 0.0,
                "noise_ratio": 0.0
            }
        
        metadata = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "rgb_path": f"rgb/{frame_id}.png",
            "depth_path": f"depth/{frame_id}.png",
            "rgb_shape": frame_rgb_saved.shape,
            "depth_shape": frame_depth_saved.shape,
            "raw_depth_shape": frame_depth.shape,
            "depth_stats": depth_stats,
            "fps": fps if fps is not None else 0.0,
            "camera_settings": {
                "rgb_resolution": "640x400",
                "depth_resolution": "640x400",
                "raw_depth_resolution": "640x400",
                "depth_align": "CAM_A",
                "median_filter": "KERNEL_7x7",
                "confidence_threshold": 255,
                "left_right_check": True,
                "subpixel": True,
                "extended_disparity": False,
                "disparity_shift": 0,
                "spatial_filter": {
                    "enable": False,
                    "hole_filling_radius": 0,
                    "num_iterations": 0
                },
                "preset_mode": "HIGH_DENSITY"
            }
        }
        
        metadata_path = os.path.join(self.annotations_dir, f"{frame_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return frame_id, metadata
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
    
    def collect_data(self, max_frames=None, save_interval=1):
        print(f"데이터 수집을 시작합니다. 저장 위치: {self.output_dir}")
        print("키보드 조작:")
        print("  - SPACE: 프레임 저장")
        print("  - 'a': 연속 저장 시작/중지")
        print("  - 'q' 또는 ESC: 종료")
        print("마우스 조작:")
        print("  - Depth 카메라 영상에 마우스를 올리면 해당 위치의 depth 값이 표시됩니다")
        
        continuous_save = False
        fps_counter = 0
        start_time = time.time()
        current_fps = 0.0
        
        try:
            with dai.Device(self.pipeline) as device:
                print("OAK-D 카메라가 성공적으로 연결되었습니다!")
                
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=True)
                
                cv2.namedWindow("OAK-D Depth Camera", cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback("OAK-D Depth Camera", self.mouse_callback)
                
                while True:
                    self.processed_count += 1  # 프레임 처리 카운트 증가
                    
                    in_rgb = q_rgb.get()
                    in_depth = q_depth.get()
                    frame_rgb = in_rgb.getCvFrame()
                    frame_depth = in_depth.getFrame()
                    
                    frame_rgb_original = frame_rgb
                    frame_depth_original = frame_depth
                    self.current_depth_frame = frame_depth
                    
                    valid_depth = frame_depth[frame_depth > 0]
                    depth_stats = {
                        "valid_pixels": int(len(valid_depth)),
                        "total_pixels": int(frame_depth.size),
                        "valid_ratio": float(len(valid_depth) / frame_depth.size) if len(valid_depth) > 0 else 0.0
                    }
                    
                    min_depth = 100
                    max_depth = 5000
                    depth_clipped = np.clip(frame_depth, min_depth, max_depth) * (frame_depth > 0)
                    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
                    depth_vis = (depth_normalized * 255).astype(np.uint8)
                    
                    frame_rgb = cv2.resize(frame_rgb, (640, 400))
                    depth_vis = cv2.resize(depth_vis, (640, 400), interpolation=cv2.INTER_NEAREST)
                    
                    depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                    if self.current_depth_frame is not None:
                        mouse_x, mouse_y = self.mouse_pos
                        if 0 <= mouse_x < 640 and 0 <= mouse_y < 400:
                            scale_x = frame_depth.shape[1] / 640
                            scale_y = frame_depth.shape[0] / 400
                            orig_x = int(mouse_x * scale_x)
                            orig_y = int(mouse_y * scale_y)
                            depth_value = frame_depth[orig_y, orig_x]
                            
                            cv2.drawMarker(depth_vis_color, (mouse_x, mouse_y), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
                            
                            if depth_value > 0 and 100 <= depth_value <= 5000:
                                depth_text = f"Depth: {depth_value}mm ({depth_value/10:.1f}cm)"
                                cv2.putText(depth_vis_color, depth_text, (mouse_x + 15, mouse_y - 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            elif depth_value > 0 and depth_value < 100:
                                depth_text = f"Close: {depth_value}mm ({depth_value/10:.1f}cm) *"
                                cv2.putText(depth_vis_color, depth_text, (mouse_x + 15, mouse_y - 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            elif depth_value > 5000:
                                depth_text = f"Far: {depth_value}mm ({depth_value/10:.1f}cm)"
                                cv2.putText(depth_vis_color, depth_text, (mouse_x + 15, mouse_y - 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            else:
                                cv2.putText(depth_vis_color, "No depth data", (mouse_x + 15, mouse_y - 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        elapsed_time = time.time() - start_time
                        current_fps = fps_counter / elapsed_time
                        fps_counter = 0
                        start_time = time.time()
                    
                    status_text = f"Frames: {self.processed_count} Valid: {depth_stats['valid_ratio']:.2f}"
                    if continuous_save:
                        status_text += " [AUTO SAVE]"
                    
                    if current_fps > 0:
                        fps_text = f"FPS: {current_fps:.1f}"
                        cv2.putText(frame_rgb, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(depth_vis_color, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.putText(frame_rgb, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(depth_vis_color, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if len(valid_depth) > 0:
                        depth_text = f"Range: 0-5000mm (0-5m, * = less accurate)"
                        cv2.putText(depth_vis_color, depth_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        cv2.putText(depth_vis_color, "No depth data", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    should_save = False
                    if continuous_save and self.processed_count % save_interval == 0:
                        should_save = True
                    
                    cv2.imshow("OAK-D RGB Camera", frame_rgb)
                    cv2.imshow("OAK-D Depth Camera", depth_vis_color)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        should_save = True
                    elif key == ord('a'):
                        continuous_save = not continuous_save
                        print(f"연속 저장: {'ON' if continuous_save else 'OFF'}")
                    elif key == ord('q') or key == 27:
                        break
                    
                    if should_save:
                        frame_id, metadata = self.save_frame_data(frame_rgb_original, frame_depth_original, current_fps)
                        print(f"저장됨: {frame_id} (FPS: {current_fps:.1f}, Valid Ratio: {metadata['depth_stats']['valid_ratio']:.2f}, Noise Ratio: {metadata['depth_stats']['noise_ratio']:.2f})")
                        self.saved_count += 1
                        
                        if max_frames and self.saved_count >= max_frames:
                            print(f"최대 프레임 수({max_frames})에 도달했습니다.")
                            break
                            
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("OAK-D 카메라가 연결되어 있는지 확인해주세요.")
        
        finally:
            cv2.destroyAllWindows()
            print(f"데이터 수집 완료. 총 {self.saved_count}개 프레임 저장됨.")
            print(f"저장 위치: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="OAK-D 카메라 데이터 수집기")
    parser.add_argument("--output", "-o", default="dataset", help="출력 디렉토리 (기본값: dataset)")
    parser.add_argument("--max-frames", "-m", type=int, help="최대 수집 프레임 수")
    parser.add_argument("--interval", "-i", type=int, default=5, help="연속 저장 간격 (기본값: 5)")
    
    args = parser.parse_args()
    
    collector = OAKDDataCollector(args.output)
    collector.collect_data(max_frames=args.max_frames, save_interval=args.interval)

if __name__ == "__main__":
    main()
