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

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="rebar_configs/franka_interface.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-position-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )

    args = parser.parse_args()
    return args


def reset_robot_poses():
    args = parse_args()

    robot_interface = FrankaInterface(
        args.interface_cfg, use_visualizer=False, automatic_gripper_reset=False
    )
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_POSITION"

    # Golden resetting joints
    # reset_joint_positions = [
    #     0.09162008114028396,
    #     -0.19826458111314524,
    #     -0.01990020486871322,
    #     -2.4732269941140346,
    #     -0.01307073642274261,
    #     2.30396583422025,
    #     0.8480939705504309,
    # ]
    # typically usage
    # reset_joint_positions = [
    #     0.084,
    #     0.41,
    #     -0.077,
    #     -2.304,
    #     -0.036,
    #     2.694,
    #     0.824,
    # ]

    # for installing ee
    reset_joint_positions = [
        0,
        0,
        0,
        -0.152,
        0,
        1.7,
        0.724,
    ]

    # For pre-tying
    # reset_joint_positions = [0.14571373489769532, -0.5633476673021741, 0.3273048401274403, -2.445724511859891, -0.2716919565525319,
    #  2.1414526547702195, 1.2490556073337815]

    # reset_joint_positions = [
    #     0.09162008114028396,
    #     -0.19826458111314524,
    #     -0.01990020486871322,
    #     -2.4732269941140346,
    #     -0.01307073642274261,
    #     2.30396583422025,
    #     0.8480939705504309,
    # ]
    # joint pos for debugging policy
  #   reset_joint_positions = [ 0.57525417, -0.55935009, -0.59125926, -2.78186322,  0.03480753,  3.65035672,
  # 0.52577797 ]


    # This is for varying initialization of joints a little bit to
    # increase data variation.
    reset_joint_positions = [
        e + np.clip(np.random.randn() * 0.001, -0.001, 0.001) # org 0.005
        for e in reset_joint_positions
    ]
    action = reset_joint_positions + [1]

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
                < 1e-3 # org 1e-3
            ):
                break


        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        print(np.array(robot_interface._state_buffer[-1].tau_J))
    robot_interface.close()


if __name__ == "__main__":
    reset_robot_poses()