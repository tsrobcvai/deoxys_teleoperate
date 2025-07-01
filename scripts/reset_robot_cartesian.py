import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from xarm.wrapper import XArmAPI

def parse_args():
    parser = argparse.ArgumentParser(description="Move xArm to a Cartesian pose for reset/init.")
    parser.add_argument('--robot', type=str, default='xarm6', help='Robot type: xarm6/lite6/uf850')
    parser.add_argument('--ip', type=str, default=None, help='xArm IP address (optional)')
    parser.add_argument('--pose', type=float, nargs=6, default=None,
                        help='Target pose [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)]')
    parser.add_argument('--wait', type=float, default=2.0, help='Wait time after move (s)')
    return parser.parse_args()

def get_default_pose(robot_type):
    """Get default reset pose based on robot type"""
    if robot_type == "lite6":
        return [210, -5, 472, -180, 0, 10]
    elif robot_type == "xarm6":
        return [481, 48.4, 315, -180, -2, 3.6]   # xarm6 pose
    elif robot_type == "uf850":
        return [300, 0, 500, -180, 0, 0] 
    else:
        return [210, -5, 472, -180, 0, 10]  # lite6 pose


def get_ip(args):
    if args.ip:
        return args.ip
    elif args.robot == "lite6":
        return "192.168.1.193"
    elif args.robot == "uf850":
        return "192.168.1.236"
    else:
        return "192.168.1.235"

def main():
    args = parse_args()
    ip = get_ip(args)
    print(f"Connecting to {args.robot} at {ip} ...")
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(0.5)

    # print("Moving to home position ...")
    # arm.move_gohome(wait=True)

    if args.pose is None:
        pose = get_default_pose(args.robot)
    else:
        pose = args.pose

    pose = np.array(pose, dtype=np.float32)
    print(f"Moving to pose: {pose} (unit: mm, deg)")
    code = arm.set_position(*pose, wait=True)
    print(f"Move result code: {code}")

    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(args.wait)

    arm.disconnect()
    print("Reset finished.")

if __name__ == "__main__":
    main()