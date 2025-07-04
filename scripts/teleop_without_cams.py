import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import cv2
import numpy as np
import json
import logging
from io_devices.meta_quest2 import Meta_quest2
from scipy.spatial.transform import Rotation
from pathlib import Path
import multiprocessing
from xarm.x3.code import APIState
import argparse
from scripts.monitor_robot_control import monitor
from io_devices.camera_redis_interface import CameraRedisSubInterface 
import yaml
from utils import YamlConfig


# Function to calculate the delta pose from the current pose to the target absolute pose
def get_delta_pose(current_pose, target_abs_pose, grasp):
    # position increment              
    current_pos = current_pose[:3, 3]
    target_pos = target_abs_pose[:3, 3]
    delta_pos = target_pos - current_pos
    # Euler angle pose increment
    current_rot = current_pose[:3, :3]
    target_rot = target_abs_pose[:3, :3]
    # relative rotation calculate
    delta_rot_mat = target_rot @ current_rot.T
    delta_rot = Rotation.from_matrix(delta_rot_mat)
    delta_euler = delta_rot.as_euler('xyz', degrees=True) 
    # grasp signal
    grasp_val = 1 if grasp else -1
    # action
    delta_action = delta_pos.tolist() + delta_euler.tolist() + [grasp_val]
    return delta_action


# Function to convert input from the device to an action for the robot arm
def input2action(device, controller_type="position_control", gripper_dof=1):
    state = device.get_controller_state()
    assert state, "please check your headset if works on debug mode correctly"
    
    # Get the target pose matrix, grasp state, reset state, action hot vector, and stop record flag
    target_pose_mat, grasp, reset, action_hot, stop_record = state["target_pose"], state["grasp"], state["reset"], state["action_hot"], state["stop_record"]

    # Get target position and rotation
    target_pos = target_pose_mat[:3, 3:]
    action_pos = target_pos.flatten() * 1

    # Get target rotation   
    target_rot = target_pose_mat[:3, :3]   # @ reset_coord_mat[:3,:3] # Reset coordiate
    target_rot_mat = Rotation.from_matrix(target_rot)
    # Convert rotation to Euler angles for xarm ufactory 
    euler_angle = target_rot_mat.as_euler('xyz', degrees=True)
    action_euler_angle = euler_angle.flatten() * 1

    action = None
    # if not reset:
    #     # grasp = 1 if grasp else -1
    #     action = action_pos.tolist() + action_euler_angle.tolist() # + [grasp]

    # return action, grasp, target_pose_mat, stop_record
    if not reset:
        if controller_type == "position_control":
            grasp_val = 1 if grasp else -1
            action = action_pos.tolist() + action_euler_angle.tolist() + [grasp_val]
        # TODO: if controller_type =="", add other controller type

    action_hot = 1 if action_hot else 0
    return action, grasp, target_pose_mat, action_hot, stop_record

    # ------ Calculate delta action (optional) # TODO: Check this step, Replay the action
    # current_pose = np.asarray(last_state.O_T_EE).reshape(4, 4).T
    # delta_action = get_delta_pose(current_pose, target_pose_mat, grasp)
    # return action, grasp, target_pose_mat, last_state, last_gripper_state, action_hot, stop_record, delta_action


# Safety control function to ensure the target action is within the robot's operational limits
def safety_control(target_action, robot_type="xarm6"):
    # unit mm
    if robot_type == "lite6":
        if not -420 <= target_action[0] <= 420:         # x axis
            return False
        if not -420 <= target_action[1] <= 420:         # y axis
            return False
        if not -150 <= target_action[2] <= 650:         # z axis
            return False
    else:  # xarm6
        if not -680 <= target_action[0] <= 680:         # x axis
            return False
        if not -680 <= target_action[1] <= 680:         # y axis  
            return False
        if not -930 <= target_action[2] <= 930:         # z axis
            return False
    return True


# Function to save images for a specific camera in a separate process
def save_images_for_camera(camera_id, img_info, img_bgr):
    """Function to save images for a specific camera in a separate process."""
    success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_bgr)
    if not success:
        print(f"Failed to save image for camera {camera_id}")

class UfactoryDataCollection():
    def __init__(self,
                 observation_cfg,
                 robot_type="xarm6",
                 controller_type="position_control",
                 folder="data",
                 max_steps=10000,
                 save2memory_first=False,
                 monitor=False,
                 #  torque_monitor=False
                 ):
        self.robot_type = robot_type
        self.folder = Path(folder)
        self.controller_type = controller_type
        self.observation_cfg = observation_cfg
        self.camera_ids = observation_cfg["camera_ids"]
        self.camera_names = observation_cfg["camera_names"]
        self.save2memory_first = save2memory_first

        self.folder.mkdir(parents=True, exist_ok=True)
        # logging config
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UfactoryDataCollection")
        self.logger.info(f"Saving to {self.folder}")
        self.max_steps = max_steps
        self.tmp_folder = None
        self.tmp_data = None
        self.robot_interface = None
        self.monitor = monitor
        # self.torque_monitor = torque_monitor
        
        # Initialize shared data structures for multiprocessing
        if self.monitor:
            manager = multiprocessing.Manager()
            self.times = manager.list()
            self.state_points = manager.list([manager.list() for _ in range(6)])  
            self.action_points = manager.list([manager.list() for _ in range(6)])
            self.monitor_save_path = str(self.folder / "robot_state_action_figure.png")
        
        # if self.torque_monitor:
        #     manager = multiprocessing.Manager()  # Create a manager for shared objects
        #     self.times = manager.list()  # Shared list for times
        #     self.tau_meansured = manager.list([manager.list() for _ in range(7)])  # Shared 6-element list for state
        #     self.tau_active = manager.list([manager.list() for _ in range(7)])  # tau_active is without gravity compensation
        #     self.monitor_save_path = "robot_tau_figure.png"

    # Process the state of the robot arm to match the expected format
    def monitor_state_process(self, mat):
        A = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        ee_data_new = np.zeros(6)
        mat = mat.reshape(4, 4, order='F')
        # To avoid singularity of axis angle representation

        rot_mat = mat[:3, :3]
        rot_mat = A[:3, :3] @ rot_mat
        # print(f"state rot_mat_after_A: {rot_mat}")
        trans_vec = mat[:3, 3]
        rot = Rotation.from_matrix(rot_mat)
        rot_vec = rot.as_rotvec() # axis angle
        ee_data_new[:3] = trans_vec
        ee_data_new[3:] = rot_vec
        # print(f"state: {ee_data_new}")
        return ee_data_new, rot_mat
   
    # Function to monitor the action process and adjust the rotation matrix
    def monitor_action_process(self, target_action):
        A = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        # print(f"before target_action: {target_action}")
        rot2 = Rotation.from_rotvec(target_action[3:6])
        rot_mat2 = rot2.as_matrix()
        rot_mat2 = A[:3,:3]  @ rot_mat2
        # print(rot_mat2)
        rot_new2 = Rotation.from_matrix(rot_mat2)
        rot_vec_new2 = rot_new2.as_rotvec()
        target_action[3:6] = rot_vec_new2
        # print(f"action: {target_action}")
        return target_action, rot_mat2
    
    # Collect data from the robot arm and save it to a folder
    def collect_data(self,ip):
        experiment_id = 0   
        for path in self.folder.glob("run*"):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split("run")[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        run_folder = self.folder / f"run{experiment_id:03d}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Initialize camera interfaces
        # cr_interfaces = {}
        # for idx, camera_id in enumerate(self.camera_ids):
        #     camera_info = {"camera_id": camera_id, "camera_name": self.camera_names[idx]}
        #     if "rs" in camera_info["camera_name"]:
        #         cr_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False) # use depth, if not false
        #     else:
        #         cr_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False)
        #     cr_interface.start()
        #     cr_interfaces[camera_id] = cr_interface

        # Initalize xarm ufactory
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        from xarm.wrapper import XArmAPI
        if self.robot_type == "lite6":
            ip = "192.168.1.193"   # lite6 ip
        elif self.robot_type == "uf850":
            ip = "192.168.1.236"   # uf850 ip
        else:
            ip = "192.168.1.235"   # xarm6 ip
        arm = XArmAPI(ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)

        # arm.move_gohome(wait=True)
        # arm.set_position(*[200, 0, 200, 180, 0, 0], wait=True) # hardcoded initial pose
        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.5)
        device = Meta_quest2()
        
        # read initial pose
        arm_init_pose = np.asarray(arm.get_position()[1])
        arm_init_pose_ = Rotation.from_euler('xyz', [arm_init_pose[3], arm_init_pose[4], arm_init_pose[5]], degrees=True)
        arm_init_pose_mat = np.eye(4)
        arm_init_pose_mat[:3, :3] = arm_init_pose_.as_matrix()
        arm_init_pose_mat[:3,  3] = arm_init_pose[:3]
        # import pdb;pdb.set_trace()
        device.start_control(arm_init_pose_mat)
        time.sleep(0.5)

        # # intial gripper
        # ret = arm.robotiq_reset()
        # code, ret = arm.robotiq_set_activate()
        # if code != 0:
        #     print('Robotiq activate failed, exit.')
        #     arm.disconnect()
        #     return
        
        data = {
            "action": [],
            "target_pose_mat": [],
            "ee_states": [],
            "joint_states": [],
            "gripper_states": [],
            "action_hot": [],
            "delta_action": [],
            "grasp": [],
        }
        for camera_id in self.camera_ids:
            data[f"camera_{camera_id}"] = []

        i = 0
        start = False
        last_grasp_state = None 

        if self.monitor:
            monitor_process = multiprocessing.Process(target=monitor, args=(self.times, self.state_points, self.action_points, self.monitor_save_path))
            monitor_process.start()

        # if self.torque_monitor:
        #     monitor_process = multiprocessing.Process(target=torque_monitor, args=(self.times, self.tau_meansured, self.tau_active, self.monitor_save_path))
        #     monitor_process.start()
        
        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()
            action, grasp, target_pose_mat, action_hot, stop_record = input2action(device=device) # , delta_action # (optional)

            if action is None and start:
                device.stop_control()
                break
            start = True

            # Safety control 
            if not safety_control(action):
                device.stop_control()
                print("The ee location is beyond the predefined safety range")
                break

            # robot arm control
            ret = arm.set_servo_cartesian(action, speed=20, mvacc=200) # action is absolute pose [x, y, z, roll, pitch, yaw]
            # print(action)

            # camera data collection
            # for camera_id in self.camera_ids:
            #     img_info = cr_interfaces[camera_id].get_img_info()
            #     imgs_array = cr_interfaces[camera_id].get_img()
                
            #     # 处理彩色图像
            #     if cr_interfaces[camera_id].use_color:
            #         img_bgr = cv2.cvtColor(imgs_array["color"], cv2.COLOR_RGB2BGR)
            #         if self.save2memory_first:
            #             img_info["color_image_data"] = img_bgr
            #         else:
            #             success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_bgr)
            #             if not success:
            #                 print(f"Failed to save image for camera {camera_id}")
                    
            #         # 保存图像信息（不是文件名）
            #         data[f"camera_{camera_id}"].append(img_info)
            #     # todo: add depth image support
            #     if cr_interfaces[camera_id].use_depth and "depth" in imgs_array:
            #         try:
            #             if self.save2memory_first:
            #                 img_info["depth_image_data"] = imgs_array["depth"]
            #             else:
            #                 success = cv2.imwrite(img_info["depth_img_name"] + ".png", imgs_array["depth"])
            #                 if not success:
            #                     print(f"failed saving depth image for camera {camera_id}")
            #         except Exception as e:
            #             print(f"Error saving depth image for camera {camera_id}: {e}")

            # robotiq gripper control，only for xarm6 
            if self.robot_type == "xarm6":
                if last_grasp_state != grasp:
                    if grasp:
                        code, ret = arm.robotiq_close(wait=False)
                        print('robotiq_close, code={}, ret={}'.format(code, ret))
                    else:
                        code, ret = arm.robotiq_open(wait=False)
                        print('robotiq_open, code={}, ret={}'.format(code, ret))
                    last_grasp_state = grasp
                    
                    if code == APIState.END_EFFECTOR_HAS_FAULT:
                        print('robotiq fault code: {}'.format(arm.robotiq_status['gFLT']))
                        device.stop_control()
                        break
            elif self.robot_type == "lite6":
                # suction gripper only for lite6, suction=close，release=open
                if last_grasp_state != grasp:
                    if grasp:
                        arm.open_lite6_gripper()   # suction
                        print('suction: open')
                    else:
                        arm.close_lite6_gripper()  # release
                        print('suction: close')
                    last_grasp_state = grasp
            
            # Collect data for monitoring
            ee_state = arm.get_position()[1]
            joint_state = arm.get_servo_angle()[1] if hasattr(arm, "get_servo_angle") else [0]*6
            gripper_state = 0  
            delta_action = []  # Optional, can be used for delta action calculation
            
            data["action"].append(action)
            data["target_pose_mat"].append(target_pose_mat)
            data["ee_states"].append(ee_state)
            data["joint_states"].append(joint_state)
            data["gripper_states"].append(gripper_state)
            data["action_hot"].append(action_hot)
            data["delta_action"].append(delta_action)
            data["grasp"].append(grasp)

            if stop_record:
                device.stop_control()
                break

            # control frequency 50 HZ (xarm ufactory 250HZ)
            elapsed_time = (time.time_ns() - start_time) / 1e9
            if elapsed_time < 0.02: # 50Hz
                time.sleep(0.02 - elapsed_time)
            # print(i)

            # Monitor the robot state and action
            if self.save2memory_first:
                print("------------- recording over, saving images... -------------")
                for camera_id in self.camera_ids:
                    for img_info in data[f"camera_{camera_id}"]:
                        if "color_image_data" in img_info:
                            success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_info["color_image_data"])
                            if not success:
                                print("failed saving imgs")
                            del img_info["color_image_data"]
                #         # Save depth image if available
                #         if "depth_image_data" in img_info:
                #             try:
                #                 success = cv2.imwrite(img_info["depth_img_name"] + ".png", img_info["depth_image_data"])
                #                 if not success:
                #                     print(f"failed saving depth image for camera {camera_id}")
                #                 del img_info["depth_image_data"]
                #             except Exception as e:
                #                 print(f"Error saving depth image for camera {camera_id}: {e}")
                # print("------------- images saved -------------")

        arm.disconnect()
        self.tmp_folder = str(run_folder)
        self.tmp_data = data
        del device
        return data, run_folder

    # Save the collected data to a temporary folder
    def save(self, keep=True, keyboard_ask=False):
        data = self.tmp_data
        os.makedirs(self.tmp_folder, exist_ok=True)
        
        with open(f"{self.tmp_folder}/config.json", "w") as f:
            config_dict = {
                "controller_type": self.controller_type,
                "observation_cfg": self.observation_cfg,
            }
            json.dump(config_dict, f)

        np.savez(f"{self.tmp_folder}/demo_action", data=np.array(data["action"]))
        np.savez(f"{self.tmp_folder}/demo_ee_states", data=np.array(data["ee_states"]))
        np.savez(f"{self.tmp_folder}/demo_target_pose_mat", data=np.array(data["target_pose_mat"]))
        np.savez(f"{self.tmp_folder}/demo_joint_states", data=np.array(data["joint_states"]))
        np.savez(f"{self.tmp_folder}/demo_gripper_states", data=np.array(data["gripper_states"]))
        np.savez(f"{self.tmp_folder}/demo_action_hot", data=np.array(data["action_hot"]))
        for camera_id in self.camera_ids:
            np.savez(f"{self.tmp_folder}/demo_camera_{camera_id}", data=np.array(data[f"camera_{camera_id}"]))

        print("Total length of the trajectory: ", len(data["action"]))
        
        if keyboard_ask:
            valid_input = False
            while not valid_input:
                try:
                    keep = input(f"Save to {self.tmp_folder} or not? (enter 0 or 1): ")
                    keep = bool(int(keep))
                    valid_input = True
                except Exception as e:
                    print("Invalid input, please enter 0 or 1 and press Enter. Error message：", e)

        if not keep:
            import shutil
            shutil.rmtree(f"{self.tmp_folder}")

        self.tmp_folder = None
        self.tmp_data = None
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default="demos_collected")
    parser.add_argument('--obs-cfg', type=str, default="configs/real_robot_observation_cfg.yml")
    parser.add_argument('--robot', type=str, default='xarm6')
    parser.add_argument('--ip', type=str, default='192.168.1.235')  # 移除 required=True，添加默认值
    parser.add_argument('--controller-type', type=str, default="position_control")
    parser.add_argument('--max-steps', type=int, default=15000)
    parser.add_argument('--save2memory-first', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    
    return parser.parse_args()

def main():
    args = parse_args()
    observation_cfg = YamlConfig(args.obs_cfg).config  
    data_collection = UfactoryDataCollection(
        observation_cfg=observation_cfg,
        robot_type=args.robot,
        controller_type=args.controller_type,
        folder=args.dataset_name,
        max_steps=args.max_steps,
        save2memory_first=args.save2memory_first,
        monitor=args.monitor
    )
    data_collection.collect_data(ip=args.ip)
    data_collection.save(keep=True, keyboard_ask=True)

if __name__ == "__main__":
    main()