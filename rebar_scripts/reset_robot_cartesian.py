"""Moving robot joint positions to initial pose for starting new experiments."""
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from scipy.spatial.transform import Rotation
import copy
import roboticstoolbox as rtb

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="rebar_configs/franka_interface.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="deoxys_control/deoxys/config/joint-position-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )

    args = parser.parse_args()
    return args

def get_pose(Pose_Mat):
    pos = Pose_Mat[:3, 3:]
    rot = Pose_Mat[:3, :3]

    R_matrix = Rotation.from_matrix(rot)
    T_axis_angle = R_matrix.as_rotvec()

    current_action = np.zeros(6)
    current_action[:3] = pos[:, 0]
    current_action[3:6] = T_axis_angle
    return current_action

def action2mat(action):

    gripper_mat = np.eye(4)
    rot = Rotation.from_rotvec(action[3:6])
    rot_mat = rot.as_matrix()
    gripper_mat[:3, :3] = rot_mat
    gripper_mat[:3, 3] = action[:3]

    return  gripper_mat
def convert_axis_angle(rot_vec, backward = False):
    if backward:
        from scipy.linalg import inv
        A = np.array(
            [[1, 0, 0, 0],
             [0, -1, 0, 0.0],
             [0, 0, -1, 0.0],
             [0, 0, 0, 1.0]])
        rot = Rotation.from_rotvec(rot_vec)
        rot_mat = rot.as_matrix()
        rot_mat_new =  rot_mat @ inv(A[:3, :3]) # A.T
        rot_new = Rotation.from_matrix(rot_mat_new)
        rot_vec_new = rot_new.as_rotvec()  # axis angle
    # else:
    #     A = np.array(
    #         [[1, 0, 0, 0],
    #          [0, -1, 0, 0.0],
    #          [0, 0, -1, 0.0],
    #          [0, 0, 0, 1.0]])
    #     rot = Rotation.from_rotvec(rot_vec)
    #     rot_mat = rot.as_matrix()
    #     rot_mat_new = A[:3, :3] @ rot_mat  # A.T
    #     rot_new = Rotation.from_matrix(rot_mat_new)
    #     rot_vec_new = rot_new.as_rotvec()  # axis angle
    return rot_vec_new
def reset_robot_poses():
    args = parse_args()

    robot_interface = FrankaInterface(
        args.interface_cfg, use_visualizer=False, automatic_gripper_reset=False
    )
    controller_cfg = YamlConfig(f"{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_POSITION"
    # reset_poses = [np.array([0.3590504,  0.18076851, 0.3766974997045057, -0.4874981, 2.4202216, -1.5599284]),
    #                np.array([0.3590504,  0.18076851, 0.3766974997045057, -0.4874981, 2.4202216, -1.5599284])]

    # reset_poses = [np.array([0.55, -0.090, 0.20, 3.04, -0.1, -0.1]), np.array([0.58, -0.1, 0.23, 3.24, 0.1, 0.1])]
    #
    # reset_poses = [np.array([0.40, 0, 0.35, 3.1415926, 0, 0]), # -0.15 to 0.15    2.84 - 3.44  -1
    #                np.array([0.40, 0, 0.35, 3.1415926, 0, 0])]

    # reset_poses = [np.array([0.55, -0.145, 0.20, 3.04, -0.1, -0.1]), np.array([0.58, -0.145, 0.23, 3.24, 0.1, 0.1])]
    # reset_poses = [np.array([0.58, -0.095, 0.20, 3.14, 0, 0]), np.array([0.58, -0.095, 0.20, 3.14, 0, 0])] # 0.1 = 5.73 degree

    # SANITY TEST: Grasp Rebar
    # reset_poses = [np.array([0.5, 0, 0.375, 0, 0, 0]),  # -0.15 to 0.15    2.84 - 3.44  -1
    #                np.array([0.5, 0, 0.375, 0, 0, 0])]

    reset_poses = [np.array([0.4, -0.1, 0.35, -0.3, -0.3, -0.3]),  # -0.15 to 0.15    2.84 - 3.44  -1
                   np.array([0.6, 0.1, 0.40, 0.3, 0.3, 0.3])]

    """ Validation set (Failure set):
    Blue marker position with vertical grasp pose
    Tend to the right
    0.57247416 -0.09898141  0.21251821  3.04455107 -0.05613548  0.09178649
    
    Tend to the left
    5.67625533e-01 -7.33828638e-02  2.03818977e-01 -5.44272736e-02 -6.88778386e-02  1.82165317e-02
    5.64615767e-01 -9.08978267e-02  2.20658140e-01  4.75879312e-02 3.43541242e-03 -1.55861843e-02
   
    x value far away from the slot
    0.52993956 -0.08730665  0.20187766 -0.0397725   0.00347194 -0.01389885
    
    grasp pose tend to the right, blue marker, it works not good
    """

    alpha = np.random.uniform(0, 1, size=reset_poses[0].shape)
    print("Alpha values:", alpha)
    # Calculate random_pose using a different alpha for each element
    random_pose = (1 - alpha) * reset_poses[0] + alpha * reset_poses[1]
    action = random_pose
    action[3:6] = convert_axis_angle(action[3:6], backward = True)

    print("Reset action:", action)
    robot = rtb.models.Panda()
    gripper_pose_mat = action2mat(action)
    reset_joint_positions = list(robot.ik_LM(gripper_pose_mat, q0=robot_interface.last_q)[0])
    action_joints = reset_joint_positions + [-1] # 0 means close the gripper, -1 means open the gripper

    while True:
        if len(robot_interface._state_buffer) > 0:
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

            if (
                    np.max(
                        np.abs(
                            np.array(robot_interface._state_buffer[-1].q)
                            - np.array(reset_joint_positions)
                        )
                    )
                    < 5e-3  # org 1e-3
            ):
                break
        robot_interface.control(
            controller_type=controller_type,
            action = action_joints,
            controller_cfg=controller_cfg,
        )
    robot_interface.close()


if __name__ == "__main__":
    reset_robot_poses()
