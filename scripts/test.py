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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--robot', type=str, default='xarm6')
parser.add_argument('--ip', type=str, required=True)
args = parser.parse_args()

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


# Function to convert input from the device to an action for the robot arm
def input2action(device):
    state = device.get_controller_state()
    assert state, "please check your headset if works on debug mode correctly"

    target_pose_mat, grasp, reset, _, stop_record = state["target_pose"], state["grasp"], state["reset"], state["action_hot"], state["stop_record"]

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
    if not reset:
        # grasp = 1 if grasp else -1
        action = action_pos.tolist() + action_euler_angle.tolist() # + [grasp]

    return action, grasp, target_pose_mat, stop_record

class UfactoryDataCollection():
    def __init__(self,
                 folder="data",
                #  observation_cfg,
                #  interface_cfg="rebar_configs/franka_interface.yml",
                #  controller_cfg="rebar_configs/osc-pose-controller.yml",
                #  controller_type="OSC_POSE",
                max_steps=10000,
                #  save2memory_first=False,
                monitor=False,
                #  torque_monitor=False
                robot_type="xarm6" 
                 ):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.monitor = monitor
        self.robot_type = robot_type 
        
        # logging config
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UfactoryDataCollection")
        self.logger.info(f"Saving to {self.folder}")
        
        if self.monitor:
            manager = multiprocessing.Manager()
            self.times = manager.list()
            self.state_points = manager.list([manager.list() for _ in range(6)])  
            self.action_points = manager.list([manager.list() for _ in range(6)])
            self.monitor_save_path = str(self.folder / "robot_state_action_figure.png")
    
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

        # Initalize xarm ufactory
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        from xarm.wrapper import XArmAPI

        #######################################################
        """
        Just for test example
        """
        # if len(sys.argv) >= 2:
        #     ip = sys.argv[1]
        # else:
        #     try:
        #         from configparser import ConfigParser
        #         parser = ConfigParser()
        #         parser.read('../robot.conf')
        #         ip = parser.get('xArm', 'ip')
        #     except:
        #         ip = input('Please input the xArm ip address:')
        #         if not ip:
        #             print('input error, exit')
        #             sys.exit(1)
        ########################################################
      
        if self.robot_type == "lite6":
            ip = "192.168.1.193"   # lite6 ip
        else:
            ip = "192.168.1.235" # xarm6 ip
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
        
        data = {"action": [],"ee_states": [],
            "target_pose_mat": [],"grasp": [],}

        i = 0
        start = False
        last_grasp_state = None 

        
        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()
            action, grasp, target_pose_mat, stop_record = input2action(device=device)
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
            print(action)

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

            # collect ee state
            ee_state = arm.get_position()[1]
            data["action"].append(action)
            data["ee_states"].append(ee_state)
            data["target_pose_mat"].append(target_pose_mat)
            data["grasp"].append(grasp)
            print(f"Step {i}, action: {action}, grasp: {grasp}, stop_record: {stop_record}")

            # monitor data collection
            if self.monitor:
                now = time.time()
                self.times.append(now)
                # current state of ee（ get_position return [0, [x, y, z, roll, pitch, yaw]])
                ee_state = arm.get_position()[1]
                ee_pose_mat = np.eye(4)
                ee_pose_mat[:3, :3] = Rotation.from_euler('xyz', ee_state[3:], degrees=True).as_matrix()
                ee_pose_mat[:3, 3] = ee_state[:3]
                ee_data_new, _ = self.monitor_state_process(ee_pose_mat)
                for idx in range(6):
                    self.state_points[idx].append(ee_state[idx])
                    self.action_points[idx].append(action[idx])
                    print("action len:", len(data["action"]))
                    print("ee_states len:", len(data["ee_states"]))
            if stop_record:
                device.stop_control()
                break

            # control frequency 50 HZ
            elapsed_time = (time.time_ns() - start_time) / 1e9
            if elapsed_time < 0.02: # 50Hz
                time.sleep(0.02 - elapsed_time)
            # print(i)

        arm.disconnect()
        self.tmp_folder = str(run_folder)
        self.tmp_data = data
        del device
        return data, run_folder

    def save(self, keep=True, keyboard_ask=False):
        data = self.tmp_data
        os.makedirs(self.tmp_folder, exist_ok=True)
        
        with open(f"{self.tmp_folder}/config.json", "w") as f:
            config_dict = {
                "controller_type": "cartesian_pose"
            }
            json.dump(config_dict, f)

        np.savez(f"{self.tmp_folder}/demo_action", data=np.array(data["action"]))
        np.savez(f"{self.tmp_folder}/demo_ee_states", data=np.array(data["ee_states"]))
        np.savez(f"{self.tmp_folder}/demo_target_pose_mat", data=np.array(data["target_pose_mat"]))
        np.savez(f"{self.tmp_folder}/demo_grasp", data=np.array(data["grasp"]))

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

def main():
    data_collection = UfactoryDataCollection(max_steps=1000, robot_type=args.robot)
    data_collection.collect_data(ip=args.ip)
    data_collection.save(keep=True, keyboard_ask=True)

if __name__ == "__main__":
    main()