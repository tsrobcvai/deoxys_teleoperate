"""This is a script to record joints by moving robot around, and save it to a list"""
import os
import time
import numpy as np
import simplejson as json
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import Meta_quest2
from scipy.spatial.transform import Rotation

config_folder = "/home/autostruct-1/workspace/rebar_policy_git/rebar_scripts/pose_estimation/eyeinhand_base/cali_config"
os.makedirs(os.path.join(config_folder), exist_ok=True)

def input2action(device, interface, controller_type="OSC_POSE", gripper_dof=1):
    state = device.get_controller_state()
    assert state, "please check your headset if works on debug mode correctly"

    target_pose_mat, grasp, reset, stop_record = state["target_pose"], state["grasp"], state["reset"], state["stop_record"]

    last_state = interface._state_buffer[-1]

    # We dont need gripper here
    # last_gripper_state = interface._gripper_state_buffer[-1]
    last_gripper_state = None

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

    return action, grasp, target_pose_mat, last_state, last_gripper_state, stop_record

def main():
    device = Meta_quest2()
    # print(config_root)
    robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)
    controller_cfg = YamlConfig(config_root + "/compliant-joint-impedance-controller.yml").as_easydict()
    controller_type = "JOINT_IMPEDANCE"

    # # Make it low impedance so that we can easily move the arm around
    # controller_cfg["Kp"]["translation"] = 50
    # controller_cfg["Kp"]["rotation"] = 50

    joints = []

    recorded_joint = False
    time.sleep(1.)
    # reset_coord_mat used to reset coordinate
    device.start_control(np.asarray(robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).T)  # @ reset_coord_mat
    time.sleep(1.)
    while True:
        spacemouse_action, grasp, _, _, _, _ = input2action(
            device=device,
            interface=robot_interface,
            controller_type="OSC_POSE",
        )

        if spacemouse_action is None:
            break

        if len(robot_interface._state_buffer) > 0:

            if spacemouse_action[-1] > 0 and not recorded_joint:
                joints.append(robot_interface._state_buffer[-1].q)
                print(len(robot_interface._state_buffer[-1].q))
                recorded_joint = True
                for _ in range(5):
                    spacemouse_action, grasp, _, _, _, _ = input2action(
                        device=device,
                        interface=robot_interface,
                        controller_type=controller_type,
                    )
            elif spacemouse_action[-1] < 0:
                recorded_joint = False
                # print("not action input")
        else:
            continue
        action = list(robot_interface._state_buffer[-1].q) + [-1]
        robot_interface.control(
            controller_type=controller_type, action=action, controller_cfg=controller_cfg
        )

    save_joints = []
    for joint in joints:
        if np.linalg.norm(joint) < 1.0:
            continue
        # print(joint)
        save_joints.append(np.array(joint).tolist())

    while True:
        try:
            save = int(input("save or not? (1 - Yes, 0 - No)"))
        except ValueError:
            print("Please input 1 or 0!")
            continue
        break

    if save:
        file_name = input("Filename to save the joints: ")
        joint_info_json_filename = f"{config_folder}/{file_name}.json"

        with open(joint_info_json_filename, "w") as f:
            json.dump({"joints": save_joints}, f, indent=4)
        print(f"Saving to {file_name}.json")

    robot_interface.close()


if __name__ == "__main__":
    main()
