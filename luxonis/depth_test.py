import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth_frame = param
        depth_value = depth_frame[y, x]
        if depth_value > 0:
            print(f"Position ({x}, {y}): Depth = {depth_value} mm ({depth_value / 10:.1f} cm)")
        else:
            print(f"Position ({x}, {y}): No depth data")

def visualize_depth(depth_frame):
    min_depth = 100
    max_depth = 5000
    depth_clipped = np.clip(depth_frame, min_depth, max_depth) * (depth_frame > 0)
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_vis = (depth_normalized * 255).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis_color

def main(depth_path="dataset/depth/frame0.png"):
    # Depth 이미지 불러오기 (uint16)
    depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_frame is None:
        print(f"Error: Could not load image from {depth_path}")
        return

    print(f"Loaded depth image: {depth_frame.shape}, dtype: {depth_frame.dtype}")

    # 시각화 이미지 생성
    depth_vis_color = visualize_depth(depth_frame)

    # 창 생성 및 마우스 콜백 설정
    cv2.namedWindow("Depth Viewer")
    cv2.setMouseCallback("Depth Viewer", mouse_callback, depth_frame)

    print("Instructions:")
    print("- Move mouse over the image to see depth values in console.")
    print("- Press 'q' or ESC to exit.")

    while True:
        cv2.imshow("Depth Viewer", depth_vis_color)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()