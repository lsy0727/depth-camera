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
        
        # 디렉토리 생성
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        self.frame_count = 0
        self.current_depth_frame = None
        self.mouse_pos = (0, 0)
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """OAK-D 파이프라인 설정 (720p 해상도, 성능 최적화)"""
        self.pipeline = dai.Pipeline()
        
        # 컬러 카메라 설정 (640x400 해상도)
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(640, 400)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        # 스테레오 깊이 카메라 설정
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        # High accuracy 설정을 위한 개별 설정들
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)
        self.stereo.setExtendedDisparity(False)
        
        # 모노 카메라 설정 (400p 해상도)
        self.mono_left = self.pipeline.create(dai.node.MonoCamera)
        self.mono_right = self.pipeline.create(dai.node.MonoCamera)
        self.mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        self.mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        
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
        """프레임 데이터 저장 (향상된 메타데이터 포함)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
        frame_id = f"frame_{self.frame_count:06d}_{timestamp}"
        
        # RGB 이미지 저장
        rgb_path = os.path.join(self.rgb_dir, f"{frame_id}.jpg")
        cv2.imwrite(rgb_path, frame_rgb)
        
        # Depth 이미지 저장 (16비트 PNG로 저장)
        depth_path = os.path.join(self.depth_dir, f"{frame_id}.png")
        cv2.imwrite(depth_path, frame_depth.astype(np.uint16))
        
        # 깊이 데이터 분석
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
        
        # 메타데이터 저장
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
                "depth_resolution": "400p",
                "depth_align": "CAM_A",
                "median_filter": "KERNEL_7x7",
                "left_right_check": True,
                "subpixel": True
            }
        }
        
        metadata_path = os.path.join(self.annotations_dir, f"{frame_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.frame_count += 1
        return frame_id, metadata
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 함수 - depth 값 표시"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
    
    def collect_data(self, max_frames=None, save_interval=1):
        """데이터 수집 시작 (고성능 최적화 버전)"""
        print(f"데이터 수집을 시작합니다. 저장 위치: {self.output_dir}")
        print("키보드 조작:")
        print("  - SPACE: 프레임 저장")
        print("  - 'a': 연속 저장 시작/중지")
        print("  - 'q' 또는 ESC: 종료")
        print("마우스 조작:")
        print("  - Depth 카메라 영상에 마우스를 올리면 해당 위치의 depth 값이 표시됩니다")
        
        continuous_save = False
        
        # FPS 계산을 위한 변수
        fps_counter = 0
        start_time = time.time()
        current_fps = 0.0
        
        try:
            with dai.Device(self.pipeline) as device:
                print("OAK-D 카메라가 성공적으로 연결되었습니다!")
                
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=True)
                
                # 마우스 콜백 설정
                cv2.namedWindow("OAK-D Depth Camera", cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback("OAK-D Depth Camera", self.mouse_callback)
                
                while True:
                    # 프레임 가져오기 (blocking=True)
                    in_rgb = q_rgb.get()
                    in_depth = q_depth.get()
                    
                    frame_rgb = in_rgb.getCvFrame()
                    frame_depth = in_depth.getFrame()
                    
                    # 현재 depth 프레임 저장 (마우스 콜백용)
                    self.current_depth_frame = frame_depth.copy()
                    
                    # 깊이 데이터 분석 (성능 최적화를 위해 주석 처리)
                    valid_depth = frame_depth[frame_depth > 0]
                    
                    # 고성능 깊이 시각화 (FPS 최적화)
                    # 모든 depth 값을 표시하되, 정확도에 따라 구분
                    min_depth = 100     # 10cm
                    max_depth = 5000  # 5m
                    
                    # 빠른 깊이 처리
                    depth_clipped = np.clip(frame_depth, min_depth, max_depth)
                    depth_clipped[frame_depth == 0] = 0
                    
                    # 빠른 정규화
                    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
                    depth_normalized = np.clip(depth_normalized, 0, 1)
                    depth_8bit = (depth_normalized * 255).astype(np.uint8)
                    
                    # JET 컬러맵 적용
                    depth_vis = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                    
                    # 0값 영역을 검은색으로 설정
                    depth_vis[frame_depth == 0] = [0, 0, 0]
                    
                    # 마우스 위치의 depth 값 표시
                    if self.current_depth_frame is not None:
                        mouse_x, mouse_y = self.mouse_pos
                        if (0 <= mouse_x < frame_depth.shape[1] and 
                            0 <= mouse_y < frame_depth.shape[0]):
                            depth_value = frame_depth[mouse_y, mouse_x]
                            
                            # 마우스 위치에 십자가 표시
                            cv2.drawMarker(depth_vis, (mouse_x, mouse_y), (255, 255, 255), 
                                         cv2.MARKER_CROSS, 10, 2)
                            
                            # depth 값 유효성 검사 (정확도에 따른 구분)
                            if depth_value > 0 and 100 <= depth_value <= 5000:
                                # 정확한 측정 범위
                                depth_text = f"Depth: {depth_value}mm ({depth_value/10:.1f}cm)"
                                cv2.putText(depth_vis, depth_text, (mouse_x + 15, mouse_y - 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            elif depth_value > 0 and depth_value < 100:
                                # 가까운 거리 (부정확할 수 있음)
                                depth_text = f"Close: {depth_value}mm ({depth_value/10:.1f}cm) *"
                                cv2.putText(depth_vis, depth_text, (mouse_x + 15, mouse_y - 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            elif depth_value > 5000:
                                # 너무 먼 거리
                                depth_text = f"Far: {depth_value}mm ({depth_value/10:.1f}cm)"
                                cv2.putText(depth_vis, depth_text, (mouse_x + 15, mouse_y - 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            else:
                                # 측정 불가
                                cv2.putText(depth_vis, "No depth data", (mouse_x + 15, mouse_y - 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # FPS 계산 및 표시 (최적화)
                    fps_counter += 1
                    if fps_counter % 15 == 0:  # 15프레임마다 FPS 업데이트 (성능 향상)
                        elapsed_time = time.time() - start_time
                        current_fps = fps_counter / elapsed_time
                        fps_counter = 0
                        start_time = time.time()
                    
                    # 상태 정보 표시
                    status_text = f"Frames: {self.frame_count}"
                    if continuous_save:
                        status_text += " [AUTO SAVE]"
                    
                    # FPS 표시
                    if current_fps > 0:
                        fps_text = f"FPS: {current_fps:.1f}"
                        cv2.putText(frame_rgb, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(depth_vis, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 프레임 수 표시
                    cv2.putText(frame_rgb, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(depth_vis, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 깊이 범위 정보 표시
                    if len(valid_depth) > 0:
                        depth_text = f"Range: 0-{max_depth}mm (0-5m, * = less accurate)"
                        cv2.putText(depth_vis, depth_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        cv2.putText(depth_vis, "No depth data", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    
                    # 연속 저장 또는 수동 저장
                    should_save = False
                    if continuous_save and self.frame_count % save_interval == 0:
                        should_save = True
                    
                    # 윈도우에 표시
                    cv2.imshow("OAK-D RGB Camera", frame_rgb)
                    cv2.imshow("OAK-D Depth Camera", depth_vis)
                    
                    # 키 입력 처리 (성능 최적화)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # SPACE
                        should_save = True
                    elif key == ord('a'):  # 'a'
                        continuous_save = not continuous_save
                        print(f"연속 저장: {'ON' if continuous_save else 'OFF'}")
                    elif key == ord('q') or key == 27:  # 'q' 또는 ESC
                        break
                    
                    # 프레임 저장
                    if should_save:
                        frame_id, metadata = self.save_frame_data(frame_rgb, frame_depth, current_fps)
                        print(f"저장됨: {frame_id} (FPS: {current_fps:.1f})")
                        
                        # 최대 프레임 수 확인
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
