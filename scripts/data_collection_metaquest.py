import os
import sys
import time
import numpy as np
from io_devices import Meta_quest2
from scipy.spatial.transform import Rotation

def input2action(device):
    state = device.get_controller_state()

    assert state, "please check your headset if works on debug mode correctly"

    target_pose_mat, grasp, reset, _, stop_record = state["target_pose"], state["grasp"], state["reset"], state["action_hot"], state["stop_record"]

    target_pos = target_pose_mat[:3, 3:]
    action_pos = target_pos.flatten() * 1
    
    target_rot = target_pose_mat[:3, :3]   # @ reset_coord_mat[:3,:3] # Reset coordiate
    target_rot_mat = Rotation.from_matrix(target_rot)
    # ufactory use euler angles
    euler_angle = target_rot_mat.as_euler('xyz', degrees=True)
    action_euler_angle = euler_angle.flatten() * 1

    action = None
    if not reset:
        grasp = 1 if grasp else -1
        action = action_pos.tolist() + action_euler_angle.tolist() + [grasp]

    return action, grasp, target_pose_mat, stop_record


class DeoxysDataCollection():
    def __init__(self,
                #  folder,
                #  observation_cfg,
                #  interface_cfg="rebar_configs/franka_interface.yml",
                #  controller_cfg="rebar_configs/osc-pose-controller.yml",
                #  controller_type="OSC_POSE",
                 max_steps=2000,
                #  save2memory_first=False,
                #  monitor=False,
                #  torque_monitor=False
                 ):
         self.max_steps = max_steps


    def collect_data(self):


        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        from xarm.wrapper import XArmAPI

        #######################################################
        """
        Just for test example
        """
        if len(sys.argv) >= 2:
            ip = sys.argv[1]
        else:
            try:
                from configparser import ConfigParser
                parser = ConfigParser()
                parser.read('../robot.conf')
                ip = parser.get('xArm', 'ip')
            except:
                ip = input('Please input the xArm ip address:')
                if not ip:
                    print('input error, exit')
                    sys.exit(1)
        ########################################################

        arm = XArmAPI(ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)

        arm.move_gohome(wait=True)
        arm.set_position(*[200, 0, 200, 180, 0, 0], wait=True) # hardcoded initial pose
        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.5)
        device = Meta_quest2()

        # read initial pose
        arm_init_pose = np.asarray(arm.get_position())
        arm_init_pose_ = Rotation.from_euler('xyz', [arm_init_pose[3], arm_init_pose[4], arm_init_pose[5]])
        arm_init_pose_mat = np.eye(4)
        arm_init_pose_mat[:3, :3] = arm_init_pose_.as_matrix()
        arm_init_pose_mat[:3,  3] = arm_init_pose[:3]

        device.start_control(arm_init_pose_mat)
        time.sleep(0.5)

        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()
            action, grasp, target_pose_mat, stop_record = input2action(device=device)
            print('set_servo_cartesian, ret={}'.format(ret))

            if action is None and start:
                device.stop_control()
                break

            start = True
            ret = arm.set_servo_cartesian(action, speed=100, mvacc=2000) # action is absolute pose [x, y, z, roll, pitch, yaw]

            # control frequency 10 HZ
            end_time = time.time_ns()
            if end_time - start_time > 0.1:
                time.sleep(0.1 - end_time + start_time)

        arm.disconnect()
        
def main():

    data_collection = DeoxysDataCollection(max_steps=2000)
    data_collection.collect_data()

if __name__ == "__main__":
    main()
