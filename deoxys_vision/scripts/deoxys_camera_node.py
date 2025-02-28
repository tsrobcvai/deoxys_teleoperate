import argparse
import json
import os
import struct
import time

import cv2
import init_path
import numpy as np
import redis
from easydict import EasyDict


from deoxys_vision.networking.camera_redis_interface import CameraRedisPubInterface
# from deoxys_vision.camera.k4a_interface import K4aInterface # ts: we don't use k4
from deoxys_vision.camera.gopro_interface import GoProInterface
from deoxys_vision.camera.webcam_interface import WebCamInterface
from deoxys_vision.camera.rs_interface import RSInterface
from deoxys_vision.utils.img_utils import preprocess_color, preprocess_depth
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="10.50.193.156")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--camera-ref", type=str)
    parser.add_argument("--camera-address", default="10.50.193.156", type=str)

    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--use-rec", action="store_true")

    parser.add_argument("--img-w", default=640, type=int)
    parser.add_argument("--img-h", default=480, type=int)
    parser.add_argument("--fps", default=30, type=int)

    parser.add_argument("--rgb-convention", default="rgb", choices=["bgr", "rgb"])

    # parser.add_argument("--no-color", action="store_true")

    # parser.add_argument("--no-depth", action="store_true")

    parser.add_argument("--visualization", action="store_true")

    parser.add_argument("--depth-visualization", action="store_true")

    args = parser.parse_args()

    assert_camera_ref_convention(args.camera_ref)
    camera_info = get_camera_info(args.camera_ref)
    # print information about the cameras to run
    camera_address = args.camera_address
    camera_config = EasyDict(
        camera_type=camera_info.camera_type,
        camera_id=camera_info.camera_id,
        use_rgb=args.use_rgb,
        use_depth=args.use_depth,
        use_rec=args.use_rec,
        rgb_convention=args.rgb_convention,
    )
    print(f"This node runs with the camera {camera_config.camera_type} with id {camera_config.camera_id}")
    print("The node will publish the following data:")
    if args.use_rgb:
        print("- Color image")
    if args.use_depth:
        print("- Depth image")
    if args.use_rec:
        print("Note that Images are rectified with undistortion")

    camera_id = camera_info.camera_id

    host = args.host
    port = args.port
    camera2redis_pub_interface = CameraRedisPubInterface(
        camera_info=camera_info,
        redis_host=host, redis_port=port, 
    )
    # Check redis if the camera id is occupied or not.
    camera_interface = None

    node_config = EasyDict(use_color=True, use_depth=True)
    if not args.use_rgb:
        node_config.use_color = False

    if not args.use_depth:
        node_config.use_depth = False

    if camera_info.camera_type == "k4a":
        raise NotImplementedError
        # camera_interface = K4aInterface()
    elif camera_info.camera_type == "rs":
        import pyrealsense2 as rs
        # import pdb;pdb.set_trace()
        color_cfg = EasyDict(
            enabled=node_config.use_color, img_w=args.img_w, img_h=args.img_h, img_format=rs.format.bgr8, fps=args.fps # we meed higher fps than policy hz    org fps=60    # img_w=1280, img_h=720, img_format=rs.format.bgr8, fps=30
        )
        # TODO： This is just available when eye2hand calibration
        # color_cfg = EasyDict(
        #     enabled=node_config.use_color, img_w=1920, img_h=1080, img_format=rs.format.bgr8, fps=args.fps
        #     #we meed higher fps than policy hz    org fps=60    # img_w=1280, img_h=720, img_format=rs.format.bgr8, fps=30
        # )
        if args.use_depth:
            depth_cfg = EasyDict(
                enabled=node_config.use_depth, img_w=args.img_w, img_h=args.img_h, img_format=rs.format.z16, fps=args.fps  # we meed higher fps than policy hz    org fps=60
            )
        else:
            depth_cfg = None

        pc_cfg = EasyDict(enabled=False)
        camera_interface = RSInterface(
            device_id=camera_id, color_cfg=color_cfg, depth_cfg=depth_cfg, pc_cfg=pc_cfg
        )
    elif camera_info.camera_type == "gopro":

        color_cfg = EasyDict(
            enabled=True, img_w=args.img_w, img_h=args.img_h, img_format='bgr', fps=args.fps
            # we meed higher fps than policy hz    org fps=60    # img_w=1280, img_h=720, img_format=rs.format.bgr8, fps=30
        )

        camera_interface = GoProInterface(
            device_id=camera_id, color_cfg=color_cfg, camera_address = camera_address
        )

    elif camera_info.camera_type == "webcam":

        color_cfg = EasyDict(
            enabled=True, img_w=args.img_w, img_h=args.img_h, img_format='bgr', fps=args.fps
            # we meed higher fps than policy hz    org fps=60    # img_w=1280, img_h=720, img_format=rs.format.bgr8, fps=30
        )
        camera_interface = WebCamInterface(
            device_id=camera_id, color_cfg=color_cfg, camera_address = camera_address
        )

    camera_interface.start()
    print("Starting")
    t = time.time()
    save_dir = f"demos_collected/images/{camera_info.camera_type}_{camera2redis_pub_interface.camera_name}_{t}"
    file_ext = "jpg"
    camera_num = 0
    os.makedirs(save_dir)

    COUNT_THRESH = 5
    counter = COUNT_THRESH

    img_counter = 0
    freq = 10.0
    MAX_IMG_NUM = 653360
    while True:
        start_time = time.time_ns()

        capture = camera_interface.get_last_obs()

        if capture is None:
            continue

        t = time.time_ns()
        if capture is None:
            continue

        save_img = camera2redis_pub_interface.get_save_img_info()

        if save_img:
            counter = 0
        else:
            counter += 1
        img_info = {
            "color_img_name": "",
            "depth_img_name": "",
            "intrinsics": {
                "color": [],
                "depth": []
            }
        }
        imgs = {}
        img_info["time"] = t
        img_info["camera_type"] = camera_info.camera_type

        img_info["intrinsics"] = {}
        if node_config.use_color:
            color_img = preprocess_color(capture["color"], flip_channel=camera_config.rgb_convention == "rgb")
            color_img_name = f"{save_dir}/color_{img_counter:09d}"
            img_info["color_img_name"] = color_img_name
            # img_info["intrinsics"]["color"] = camera_interface.get_color_intrinsics(mode="dict")
            # img_info["distortion"]["color"] = camera_interface.get_color_distortion()
            # intrinsics_matrix = camera_interface.get_color_intrinsics(mode="matrix")
            # color_distortion = camera_interface.get_color_distortion()
            
            if camera_config.use_rec:
                if camera_info.camera_type == "rs":
                    if camera_id == 0:
                        color_distortion = np.array([[-3.21808020e+01],
                                            [ 1.56948008e+02],
                                            [ 1.08836334e-04],
                                            [-3.35339398e-03],
                                            [ 2.35932470e+03],
                                            [-3.23246223e+01],
                                            [ 1.62487586e+02],
                                            [ 2.30373935e+03],
                                            [ 0.00000000e+00],
                                            [ 0.00000000e+00],
                                            [ 0.00000000e+00],
                                            [ 0.00000000e+00],
                                            [ 0.00000000e+00],
                                            [ 0.00000000e+00]])
                        
                    else:
                        color_disotortion = np.array(
                            [[-1.03494286e+01],
                            [ 1.81229044e+02],
                            [-1.33669038e-03],
                            [ 3.86838065e-03],
                            [ 6.03400600e+02],
                            [-1.04039164e+01],
                            [ 1.82251593e+02],
                            [ 5.75642409e+02],
                            [ 0.00000000e+00],
                            [ 0.00000000e+00],
                            [ 0.00000000e+00],
                            [ 0.00000000e+00],
                            [ 0.00000000e+00],
                            [ 0.00000000e+00]])      
                imgs["color"] = cv2.undistort(color_img, intrinsics_matrix, color_distortion, None)
            else:
                imgs["color"] = color_img

        if node_config.use_depth:
            depth_img = preprocess_depth(capture["depth"])
            depth_img_name = f"{save_dir}/depth_{img_counter:09d}"
            img_info["depth_img_name"] = depth_img_name
            img_info["intrinsics"]["depth"] = camera_interface.get_depth_intrinsics(mode="dict")
            # depth_distortion = camera_interface.get_depth_distortion()
            intrinsics_depth_matrix = camera_interface.get_depth_intrinsics(mode="matrix")            
            imgs["depth"] = depth_img

        # print(color_img_name, ": ", img_info["time"])
        # print(img_info)
        camera2redis_pub_interface.set_img_info(img_info)
        camera2redis_pub_interface.set_img_buffer(imgs=imgs)

        if counter < COUNT_THRESH:
            img_counter += 1
            img_counter = img_counter % MAX_IMG_NUM
            # print("counter < COUNT_THRESH %s"%img_counter)
        # else:
            # print(img_counter)


        # if not args.eval:
        #     if counter < COUNT_THRESH:
        #         # Save img to tmp file
        #         if node_config.use_color:
        #             cv2.imwrite(color_img_name + ".jpg", imgs["color"])
        #         if node_config.use_depth:
        #             cv2.imwrite(depth_img_name + ".png", imgs["depth"])
        #         img_counter += 1
        # if not args.eval:
        #     # Save img to tmp file
        #     if node_config.use_color:
        #         cv2.imwrite(color_img_name, imgs["color"])
        #     if node_config.use_depth:
        #         cv2.imwrite(depth_img_name, imgs["depth"])

        # r.set(f"camera_{camera_num}::last_depth_img", depth_img.tobytes())

        if args.visualization:
            if args.rgb_convention == "rgb":
                cv2.imshow("", imgs["color"][..., ::-1])
            else:
                cv2.imshow("", imgs["color"])
            if args.depth_visualization:
                cv2.imshow("depth", imgs["depth"] * 0.001)
            cv2.waitKey(10)
        end_time = time.time_ns()

        time_interval = (end_time - start_time) / (10 ** 9)
        if time_interval < 1.0 / freq:
            time.sleep(1.0 / freq - time_interval)
            print(f"The camera node took {time_interval} to transmit image")
        else:
            print(time_interval)

        if camera2redis_pub_interface.finished:
            break


if __name__ == "__main__":
    main()
