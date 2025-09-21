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
        
        self.frame_count = 0
        self.current_depth_frame = None
        self.mouse_pos = (0, 0)
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """OAK-D 파이프라인 설정 (DepthAI 표준 Depth 계산 적용)"""
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
        self.stereo.initialConfig.setConfidenceThreshold(245)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.initialConfig.setDisparityShift(0)
        self.stereo.setOutputSize(1280, 720)
        
        # 모노 카메라 설정
        self.mono_left = self.pipeline.create(dai.node.MonoCamera)
        self.mono_right = self.pipeline.create(dai.node.MonoCamera)
        self.mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        self.mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
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
        """프레임 데이터 저장 (디스패리티 통계 포함)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_id = f"frame_{self.frame_count:06d}_{timestamp}"
        
        rgb_path = os.path.join(self.rgb_dir, f"{frame_id}.jpg")
        cv2.imwrite(rgb_path, frame_rgb)
        
        depth_path = os.path.join(self.depth_dir, f"{frame_id}.png")
        cv2.imwrite(depth_path, frame_depth.astype(np.uint16))
        
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
                "std_depth": float(valid_depth.std())
            }
        else:
            depth_stats = {
                "valid_pixels": 0,
                "total_pixels": int(frame_depth.size),
                "valid_ratio": 0.0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "mean_depth": 0.0,
                "std_depth": 0.0
            }
        
        metadata = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "rgb_path": f"rgb/{frame_id}.jpg",
            "depth_path": f"depth/{frame_id}.png",
            "rgb_shape": frame_rgb.shape,
            "depth_shape": frame_depth.shape,
            "depth_stats": depth_stats,
            "fps": fps if fps is not None else 0.0,
            "camera_settings": {
                "rgb_resolution": "640x400",
                "depth_resolution": "720p",
                "depth_align": "CAM_A",
                "median_filter": "KERNEL_7x7",
                "confidence_threshold": 245,
                "left_right_check": True,
                "subpixel": True,
                "extended_disparity": False,
                "disparity_shift": 0,
                "preset_mode": "HIGH_DENSITY"
            }
        }
        
        metadata_path = os.path.join(self.annotations_dir, f"{frame_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        self.frame_count += 1
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
                    in_rgb = q_rgb.get()
                    in_depth = q_depth.get()
                    frame_rgb = in_rgb.getCvFrame()
                    frame_depth = in_depth.getFrame()
                    
                    # 프레임 복사 최소화
                    frame_rgb_original = frame_rgb
                    frame_depth_original = frame_depth
                    self.current_depth_frame = frame_depth
                    
                    # Depth 통계 계산
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
                    depth_vis = cv2.resize(depth_vis, (640, 400))
                    
                    depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                    if self.current_depth_frame is not None:
                        mouse_x, mouse_y = self.mouse_pos
                        if 0 <= mouse_x < 640 and 0 <= mouse_y < 400:
                            # 마우스 좌표를 원본 Depth 좌표로 변환
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
                    
                    status_text = f"Frames: {self.frame_count} Valid: {depth_stats['valid_ratio']:.2f}"
                    if continuous_save:
                        status_text += " [AUTO SAVE]"
                    
                    if current_fps > 0:
                        fps_text = f"FPS: {current_fps:.1f}"
                        cv2.putText(frame_rgb, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(depth_vis_color, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.putText(frame_rgb, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(depth_vis_color, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if len(valid_depth) > 0:
                        depth_text = f"Range: 0-{max_depth}mm (0-5m, * = less accurate)"
                        cv2.putText(depth_vis_color, depth_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        cv2.putText(depth_vis_color, "No depth data", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    should_save = False
                    if continuous_save and self.frame_count % save_interval == 0:
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
                        print(f"저장됨: {frame_id} (FPS: {current_fps:.1f}, Valid Ratio: {metadata['depth_stats']['valid_ratio']:.2f})")
                        
                        if max_frames and self.frame_count >= max_frames:
                            print(f"최대 프레임 수({max_frames})에 도달했습니다.")
                            break
                            
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("OAK-D 카메라가 연결되어 있는지 확인해주세요.")
        
        finally:
            cv2.destroyAllWindows()
            print(f"데이터 수집 완료. 총 {self.frame_count}개 프레임 저장됨.")
            print(f"저장 위치: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="OAK-D 카메라 데이터 수집기")
    parser.add_argument("--output", "-o", default="dataset", help="출력 디렉토리 (기본값: dataset)")
    parser.add_argument("--max-frames", "-m", type=int, help="최대 수집 프레임 수")
    parser.add_argument("--interval", "-i", type=int, default=1, help="연속 저장 간격 (기본값: 1)")
    
    args = parser.parse_args()
    
    collector = OAKDDataCollector(args.output)
    collector.collect_data(max_frames=args.max_frames, save_interval=args.interval)

if __name__ == "__main__":
    main()
