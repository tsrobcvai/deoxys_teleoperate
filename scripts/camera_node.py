import argparse
import json
import os
import struct
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import redis
from easydict import EasyDict

from io_devices.camera_redis_interface import CameraRedisPubInterface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--camera-id", type=str, default="0")
    parser.add_argument("--camera-name", type=str, default="webcam_0")
    parser.add_argument("--camera-ref", type=str)
    parser.add_argument("--camera-address", default="/dev/video0", type=str)
    
    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument("--use-depth", action="store_true")
    
    parser.add_argument("--img-w", default=640, type=int)
    parser.add_argument("--img-h", default=480, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--use-realsense", action="store_true", help="Use RealSense SDK instead of OpenCV")
    parser.add_argument("--visualization", action="store_true")
    
    parser.add_argument("--rgb-convention", default="rgb", choices=["bgr", "rgb"])

    args = parser.parse_args()

    # 兼容两种参数方式
    if args.camera_ref:
        # 解析camera_ref格式 (如: webcam_0, rs_1, gopro_0)
        parts = args.camera_ref.split('_')
        camera_type = parts[0]
        camera_id = parts[1] if len(parts) > 1 else "0"
        camera_name = args.camera_ref
    else:
        # 使用传统参数
        camera_type = "webcam"  # 默认类型
        camera_id = args.camera_id
        camera_name = args.camera_name

    # 相机信息
    camera_info = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "camera_type": camera_type
    }

    print(f"Starting camera node for {camera_name} (ID: {camera_id}, Type: {camera_type})")
    
    # 初始化Redis发布接口
    camera2redis_pub_interface = CameraRedisPubInterface(
        camera_info=camera_info,
        redis_host=args.host, 
        redis_port=args.port
    )

    # 初始化相机
    pipeline = None
    cap = None
    
    if args.use_realsense or camera_type == "rs":
        # 使用RealSense SDK
        try:
            import pyrealsense2 as rs
            
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, args.img_w, args.img_h, rs.format.bgr8, args.fps)
            if args.use_depth:
                config.enable_stream(rs.stream.depth, args.img_w, args.img_h, rs.format.z16, args.fps)
            
            pipeline.start(config)
            print("RealSense camera initialized successfully")
            camera_type = "rs"
            
            def get_frame():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if args.use_depth else None
                
                result = {}
                if color_frame:
                    result["color"] = np.asanyarray(color_frame.get_data())
                if depth_frame:
                    result["depth"] = np.asanyarray(depth_frame.get_data())
                return result if result else None
                
        except ImportError:
            print("pyrealsense2 not installed, falling back to OpenCV")
            args.use_realsense = False
    
    if not args.use_realsense and camera_type != "rs":
        # 使用OpenCV for webcam/gopro
        device_id = int(camera_id) if camera_id.isdigit() else args.camera_address
        
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera at {device_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.img_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.img_h)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        print(f"OpenCV camera initialized successfully for {camera_type}")
        
        def get_frame():
            ret, frame = cap.read()
            return {"color": frame} if ret else None

    # 创建保存目录
    save_dir = f"demos_collected/images/{camera_name}_{int(time.time())}"
    os.makedirs(save_dir, exist_ok=True)

    img_counter = 0
    freq = 10.0  # 发布频率
    MAX_IMG_NUM = 999999
    COUNT_THRESH = 5
    counter = COUNT_THRESH

    print("Camera publisher started... Press Ctrl+C to stop")
    
    try:
        while True:
            start_time = time.time_ns()

            # 获取图像
            capture = get_frame()
            if capture is None:
                continue

            # 检查是否需要保存图像
            save_img = camera2redis_pub_interface.get_save_img_info()
            
            if save_img:
                counter = 0
            else:
                counter += 1

            # 准备图像信息
            t = time.time_ns()
            img_info = {
                "color_img_name": f"{save_dir}/color_{img_counter:09d}",
                "depth_img_name": f"{save_dir}/depth_{img_counter:09d}" if args.use_depth else "",
                "time": t,
                "camera_type": camera_type,
                "intrinsics": {"color": [], "depth": []}
            }

            # 准备图像数据
            imgs = {}
            if "color" in capture and capture["color"] is not None:
                color_img = capture["color"]
                # 根据RGB约定转换颜色
                if args.rgb_convention == "rgb":
                    imgs["color"] = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                else:
                    imgs["color"] = color_img
                    
            if "depth" in capture and capture["depth"] is not None:
                imgs["depth"] = capture["depth"]

            # 发布到Redis
            camera2redis_pub_interface.set_img_info(img_info)
            camera2redis_pub_interface.set_img_buffer(imgs=imgs)

            # 更新计数器
            if counter < COUNT_THRESH:
                img_counter = (img_counter + 1) % MAX_IMG_NUM

            # 可视化
            if args.visualization:
                if "color" in imgs:
                    display_img = imgs["color"]
                    if args.rgb_convention == "rgb":
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Camera {camera_name}", display_img)
                    
                if "depth" in imgs:
                    depth_display = (imgs["depth"] * 0.001).astype(np.float32)
                    cv2.imshow(f"Depth {camera_name}", depth_display)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 控制发布频率
            end_time = time.time_ns()
            time_interval = (end_time - start_time) / (10 ** 9)
            if time_interval < 1.0 / freq:
                time.sleep(1.0 / freq - time_interval)

            # 检查是否结束
            if camera2redis_pub_interface.finished:
                break

    except KeyboardInterrupt:
        print("\nStopping camera publisher...")
    
    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Camera publisher stopped")

if __name__ == "__main__":
    main()