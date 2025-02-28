import json
import os
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
# from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import Meta_quest2
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils import YamlConfig, transform_utils
import matplotlib.pyplot as plt # for monitoring
from rebar_scripts.monitor_robot_control import monitor, torque_monitor
from scipy.spatial.transform import Rotation
import multiprocessing
import copy

logger = get_deoxys_example_logger()

# reset_coord_mat = np.array([[1.0, 0.0, 0.0, 0.0],
#                           [0.0, -1.0, 0.0, 0.0],
#                           [0.0, 0.0, -1.0, 0.0],
#                           [0.0, 0.0, 0.0, 1.0]])
def get_delta_pose(current_pose, target_abs_pose, grasp):

    current_pos = current_pose[:3, 3:]
    current_rot = current_pose[:3, :3]
    current_quat = Rotation.from_matrix(current_rot).as_quat()
    target_pos = target_abs_pose[:3, 3:]
    target_rot = target_abs_pose[:3, :3]
    target_quat = Rotation.from_matrix(target_rot).as_quat()
    quat_diff = Rotation.from_quat(target_quat) * Rotation.from_quat(current_quat).inv()
    axis_angle_diff = quat_diff.as_rotvec()
    action_pos = (target_pos - current_pos).flatten() * 1
    action_axis_angle = axis_angle_diff.flatten() * 1
    grasp = 1 if grasp else -1
    delta_action = action_pos.tolist() + action_axis_angle.tolist() + [grasp]
    return delta_action

def input2action(device, interface, controller_type="OSC_POSE", gripper_dof=1):
    state = device.get_controller_state()
    assert state, "please check your headset if works on debug mode correctly"

    target_pose_mat, grasp, reset, action_hot, stop_record = state["target_pose"], state["grasp"], state["reset"], state["action_hot"], state["stop_record"]

    last_state = interface._state_buffer[-1]
    last_gripper_state = interface._gripper_state_buffer[-1]

    target_pos = target_pose_mat[:3, 3:]
    target_rot = target_pose_mat[:3, :3]   # @ reset_coord_mat[:3,:3] # Reset coordiate
    target_rot_mat = Rotation.from_matrix(target_rot)
    axis_angle = target_rot_mat.as_rotvec()

    action_pos = target_pos.flatten() * 1
    action_axis_angle = axis_angle.flatten() * 1

    action = None

    if not reset:
        if controller_type == "OSC_POSE":
            grasp = 1 if grasp else -1
            action = action_pos.tolist() + action_axis_angle.tolist() + [grasp]

        if controller_type == "OSC_YAW":
            action_axis_angle[:2] = 0.

            grasp = 1 if grasp else -1
            action = action_pos.tolist() + action_axis_angle.tolist() + [grasp]

        if controller_type == "OSC_POSITION":
            action_axis_angle[:3] = 0.

            grasp = 1 if grasp else -1
            action = action_pos.tolist() + action_axis_angle.tolist() + [grasp]

        if controller_type == "JOINT_IMPEDANCE":
            grasp = 1 if grasp else -1
            action = np.array([0.0] * 7 + [grasp] * gripper_dof)

    # process action_hot: 1 means recovery mode (optional)
    action_hot = 1 if action_hot else 0

    # ------ Calculate delta action (optional) # TODO: Check this step, Replay the action
    # current_pose = np.asarray(last_state.O_T_EE).reshape(4, 4).T
    # delta_action = get_delta_pose(current_pose, target_pose_mat, grasp)
    # return action, grasp, target_pose_mat, last_state, last_gripper_state, action_hot, stop_record, delta_action

    return action, grasp, target_pose_mat, last_state, last_gripper_state, action_hot, stop_record

def safety_control(target_action):
    if not 0.28 <= target_action[0] <= 0.79:
        return False
    if not -0.3 <= target_action[1] <= 0.3:
        return False
    if not 0.0 <= target_action[2]:
        return False
    return True

def save_images_for_camera(camera_id,img_info, img_bgr):
    """Function to save images for a specific camera in a separate process."""
    success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_bgr)
    if not success:
        print(f"Failed to save image for camera {camera_id}")

class DeoxysDataCollection():
    def __init__(self,
                 folder,
                 observation_cfg,
                 interface_cfg="rebar_configs/franka_interface.yml",
                 controller_cfg="rebar_configs/osc-pose-controller.yml",
                 controller_type="OSC_POSE",
                 max_steps=2000,
                 save2memory_first=False,
                 monitor=False,
                 torque_monitor=False
                 ):

        self.folder = Path(folder)
        self.interface_cfg = interface_cfg
        self.controller_cfg = controller_cfg
        # self.controller_cfg_dic = YamlConfig(self.controller_cfg).as_easydict()
        self.controller_type = controller_type
        self.observation_cfg = observation_cfg
        self.camera_ids = observation_cfg.camera_ids
        self.camera_names = observation_cfg.camera_names
        self.save2memory_first = save2memory_first

        self.folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {self.folder}")
        self.max_steps = max_steps
        self.tmp_folder = None
        self.tmp_data = None
        self.robot_interface = None
        self.monitor = monitor
        self.torque_monitor = torque_monitor

        if self.monitor:
            manager = multiprocessing.Manager()  # Create a manager for shared objects
            self.times = manager.list()  # Shared list for times
            self.state_points = manager.list([manager.list() for _ in range(6)])  # Shared 6-element list for state
            self.action_points = manager.list([manager.list() for _ in range(6)])  # Shared 6-element list for action  # Shared 6-element list for action
            self.monitor_save_path = "robot_state_action_figure.png"

        if self.torque_monitor:
            manager = multiprocessing.Manager()  # Create a manager for shared objects
            self.times = manager.list()  # Shared list for times
            self.tau_meansured = manager.list([manager.list() for _ in range(7)])  # Shared 6-element list for state
            self.tau_active = manager.list([manager.list() for _ in range(7)])  # tau_active is without gravity compensation
            self.monitor_save_path = "robot_tau_figure.png"

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
        rot_vec = rot.as_rotvec()  # axis angle
        ee_data_new[:3] = trans_vec
        ee_data_new[3:] = rot_vec
        # print(f"state: {ee_data_new}")
        return ee_data_new, rot_mat

    def monitor_action_process(self, target_action):
        A = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        # print(f"before target_action: {target_action}")
        rot2 = Rotation.from_rotvec(target_action[3:6])
        rot_mat2 = rot2.as_matrix()
        rot_mat2 = A[:3,:3]  @ rot_mat2  # A.T
        # print(rot_mat2)
        rot_new2 = Rotation.from_matrix(rot_mat2)
        rot_vec_new2 = rot_new2.as_rotvec()  # axis angle
        target_action[3:6] = rot_vec_new2
        # print(f"action: {target_action}")
        return target_action, rot_mat2

    def collect_data(self):
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
        folder = str(self.folder / f"run{experiment_id:03d}")

        self.robot_interface = FrankaInterface(self.interface_cfg, automatic_gripper_reset = False)

        cr_interfaces = {}
        for idx, camera_id in enumerate(self.camera_ids):
            camera_info = {"camera_id": camera_id,
                           "camera_name": self.camera_names[idx]}

            # cr_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False)
            if "rs" in camera_info["camera_name"]:   #TODO: hard code if record depth or not
                cr_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False) # or use_depth=True
            else:
                cr_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False)
            cr_interface.start()
            cr_interfaces[camera_id] = cr_interface

        controller_cfg = YamlConfig(
            self.controller_cfg
        ).as_easydict()

        # data = {"action": [], "action_robot_gripper": [], "target_pose_mat": [],
        #         "ee_states": [], "joint_states": [], "gripper_states": [],
        #         "action_hot": [], "delta_action": []}
        data = {"action": [], "target_pose_mat": [],
                "ee_states": [], "joint_states": [], "gripper_states": [],
                "action_hot": [], "delta_action": []}
        for camera_id in self.camera_ids:
            data[f"camera_{camera_id}"] = []

        i = 0
        start = False
        previous_state_dict = None

        time.sleep(2)
        device = Meta_quest2()
        # reset_coord_mat used to reset coordinate
        device.start_control(np.asarray(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).T) #  @ reset_coord_mat
        time.sleep(2)

        if self.monitor:
            monitor_process = multiprocessing.Process(target=monitor, args=(self.times, self.state_points, self.action_points, self.monitor_save_path))
            monitor_process.start()

        if self.torque_monitor:
            monitor_process = multiprocessing.Process(target=torque_monitor, args=(self.times, self.tau_meansured, self.tau_active, self.monitor_save_path))
            monitor_process.start()

        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()

            action, grasp, target_pose_mat, last_state, last_gripper_state, action_hot, stop_record = input2action(
                device=device,
                interface=self.robot_interface,
                controller_type=self.controller_type,
            ) # , delta_action # (optional)

            if action is None and start:
                device.stop_control()
                break

            if np.linalg.norm(action[:-1]) < 1e-3 and not start: # delta action should not be too small
                continue
            start = True

            if not stop_record:
                for camera_id in self.camera_ids:
                    # start_time_cam = time.time()
                    img_info = cr_interfaces[camera_id].get_img_info() #  aims to record image path
                    # print(img_info)0
                    imgs_array = cr_interfaces[camera_id].get_img()
                    # end_time_cam1 = time.time()
                    if cr_interfaces[camera_id].use_color:
                        img_bgr = cv2.cvtColor(imgs_array["color"], cv2.COLOR_RGB2BGR)

                        if self.save2memory_first:
                            # 2 --- saving images into memory first
                            img_info["color_image_data"] = img_bgr
                        else:
                            # 1 --- saving images into disk directly (high latency if image data is large)
                            success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_bgr)
                            if not success:
                                print("failed saving imgs")

                        data[f"camera_{camera_id}"].append(img_info)

                    if cr_interfaces[camera_id].use_depth:
                        raise NotImplementedError # "To implement correct depth resize later"
                        success = cv2.imwrite(img_info["depth_img_name"] + ".png", imgs_array["depth"])
                        if not success:
                            print("failed saving imgs")

                    # end_time_cam = time.time()
                    # print(camera_id, end_time_cam1-start_time_cam)
                    # print(camera_id, end_time_cam - start_time_cam)


            # if self.controller_type == "OSC_YAW":
            #     action[3:5] = 0.0
            # elif self.controller_type == "OSC_POSITION":
            #     action[3:6] = 0.0

            # if not safety_control(action):
            #     device.stop_control()
            #     print("The ee location is beyond the predefined safety range")
            #     break

            # action is axis angle
            import pdb; pdb.set_trace()
            
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )

            # if len(self.robot_interface._state_buffer) == 0:
            #     continue

            if not stop_record:
                # print(action, action_hot)
                target_action = np.zeros(7)
                target_action[:6] = action[:6]
                target_action[6] = grasp

                # ----------------- start saving demonstration ------------
                data["action"].append(action)
                # data["action_robot_gripper"].append(target_action) # TODO: this should be the same as "action", so it could be deleted
                data["target_pose_mat"].append(target_pose_mat)
                data["action_hot"].append(action_hot)

                # -------- (optional)
                # data["delta_action"].append(delta_action)

                # Note: To aviod singularity when using axis angles, we need to reset the end effector coordinate before training the model
                # After this, the orientation in the coordinate of ee will be line with that of robot base

                state_dict = {
                    "ee_states": np.array(last_state.O_T_EE), # 16-element vector # TODO: check if last_state varies consistantly
                    "joint_states": np.array(last_state.q),
                    "gripper_states": np.array(last_gripper_state.width),
                }
                print(np.array(last_state.q))
                if self.monitor:
                    self.times.append(i)
                    target_state_monitor = self.monitor_state_process(copy.deepcopy(np.array(last_state.O_T_EE)))[0]
                    target_action_monitor = self.monitor_action_process(copy.deepcopy(target_action[:6]))[0]
                    # K_F_ext_hat_K
                    # target_state_monitor = copy.deepcopy(np.array(last_state.K_F_ext_hat_K))
                    # target_action_monitor = copy.deepcopy(np.array(last_state.K_F_ext_hat_K))

                    for j in range(6):
                        self.state_points[j].append(target_state_monitor[j])
                        self.action_points[j].append(target_action_monitor[j])

                if self.torque_monitor:
                    self.times.append(i)
                    tau_meansured_monitor = copy.deepcopy(np.array(last_state.tau_J))
                    tau_active_monitor = copy.deepcopy(np.array(last_state.tau_J_d))
                    # Joint speed
                    # tau_meansured_monitor = copy.deepcopy(np.array(last_state.dq))
                    # tau_active_monitor = copy.deepcopy(np.array(last_state.dq_d))
                    for j in range(7):
                        self.tau_meansured[j].append(tau_meansured_monitor[j])
                        self.tau_active[j].append(tau_active_monitor[j])

                if previous_state_dict is not None:
                    for proprio_key in state_dict.keys():
                        proprio_state = state_dict[proprio_key]
                        if np.sum(np.abs(proprio_state)) <= 1e-6:
                            proprio_state = previous_state_dict[proprio_key]
                        state_dict[proprio_key] = np.copy(proprio_state)

                previous_state_dict = state_dict

                # add sate_dict into data
                for proprio_key in state_dict.keys():
                    data[proprio_key].append(state_dict[proprio_key])
            else:
                print("------------- strop recording demo, but you can still control the arm -------------")
                print("------------- press A to stop control -------------")
            end_time = time.time_ns()
            print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

        for camera_id in self.camera_ids:
            cr_interfaces[camera_id].stop()
        if self.save2memory_first:
            # ---- just be used when we want to save to memory first
            print("------------- recording over, saving images... -------------")
            for camera_id in self.camera_ids:
                for img_info in data[f"camera_{camera_id}"]:
                    # Check if color image data exists in img_info
                    if "color_image_data" in img_info:
                        success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_info["color_image_data"])
                        if not success:
                            print("failed saving imgs")
                        # (Important) Remove the in-memory image data after saving
                        del img_info["color_image_data"]
            print("------------- images saved -------------")

        self.tmp_folder = folder
        self.tmp_data = data

        del device

        return data, folder
    def save(self, keep=True, keyboard_ask=False):
        data = self.tmp_data
        os.makedirs(self.tmp_folder, exist_ok=True)
        with open(f"{self.tmp_folder}/config.json", "w") as f:
            config_dict = {
                "controller_cfg": {}, #TODO
                "controller_type": self.controller_type,
                "observation_cfg": self.observation_cfg,
            }
            json.dump(config_dict, f)

        np.savez(f"{self.tmp_folder}/testing_demo_action", data=np.array(data["action"]))
        # np.savez(f"{self.tmp_folder}/testing_demo_action_robot_gripper", data=np.array(data["action_robot_gripper"]))
        np.savez(f"{self.tmp_folder}/testing_demo_target_pose_mat", data=np.array(data["target_pose_mat"]))
        np.savez(f"{self.tmp_folder}/testing_demo_ee_states", data=np.array(data["ee_states"]))
        np.savez(f"{self.tmp_folder}/testing_demo_joint_states", data=np.array(data["joint_states"]))
        np.savez(f"{self.tmp_folder}/testing_demo_gripper_states", data=np.array(data["gripper_states"]))
        np.savez(f"{self.tmp_folder}/testing_demo_action_hot", data=np.array(data["action_hot"]))
        # -------- (optional)
        # np.savez(f"{self.tmp_folder}/testing_demo_delta_action", data=np.array(data["delta_action"]))

        for camera_id in self.camera_ids:
            np.savez(f"{self.tmp_folder}/testing_demo_camera_{camera_id}", data=np.array(data[f"camera_{camera_id}"]))

        print("Total length of the trajectory: ", len(data["action"]))

        if keyboard_ask:
            valid_input = False
            while not valid_input:
                try:
                    keep = input(f"Save to {self.tmp_folder} or not? (enter 0 or 1)")
                    keep = bool(int(keep))
                    valid_input = True
                except:
                    pass

        if not keep:
            import shutil
            shutil.rmtree(f"{self.tmp_folder}")

        self.tmp_folder = None
        self.tmp_data = None
        self.robot_interface.close()
        del self.robot_interface
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-name',
        type=str,
        default="demos_collected"
    )
    parser.add_argument(
        '--obs-cfg',
        type=str,
        default="rebar_configs/real_robot_observation_cfg.yml"
    )
    parser.add_argument(
        '--franka-cfg',
        type=str,
        default="rebar_configs/franka_interface.yml"
    )
    parser.add_argument(
        '--controller-cfg',
        type=str,
        default="rebar_configs/osc-pose-controller.yml"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    assert(args.dataset_name is not None)

    observation_cfg = YamlConfig(args.obs_cfg).as_easydict()

    data_collection = DeoxysDataCollection(
        observation_cfg=observation_cfg,
        interface_cfg=args.franka_cfg,
        controller_cfg=args.controller_cfg,
        controller_type="OSC_POSE",
        folder=args.dataset_name,
        max_steps=1500,
        save2memory_first = True,
        monitor = False,
        torque_monitor = False)

    data_collection.collect_data()
    data_collection.save(keep=True, keyboard_ask=True)

if __name__ == "__main__":
    main()
