import os
import sys
import time
import math
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from xarm.wrapper import XArmAPI

def parse_args():
    parser = argparse.ArgumentParser(description="Move xArm joints to initial pose.")
    parser.add_argument('--robot', type=str, default='xarm6', help='Robot type: xarm6/lite6/uf850')
    parser.add_argument('--ip', type=str, default=None, help='xArm IP address (optional)')
    parser.add_argument('--joint', type=float, nargs=6, default=[0, 0, 90, 0, 90 , -17],
                        help='Target joint positions (deg or rad)')
    parser.add_argument('--is-radian', action='store_true', help='Interpret joint values as radians')
    parser.add_argument('--speed', type=float, default=50, help='Joint move speed (deg/s or rad/s)')
    return parser.parse_args()

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

    joint = np.array(args.joint, dtype=np.float32)
    speed = args.speed
    if args.is_radian:
        print(f"Moving to joint (radian): {joint}, speed={speed} rad/s")
        arm.set_servo_angle(angle=joint, speed=speed, is_radian=True, wait=True)
    else:
        print(f"Moving to joint (degree): {joint}, speed={speed} deg/s")
        arm.set_servo_angle(angle=joint, speed=speed, is_radian=False, wait=True)
    print("Current joint (deg):", arm.get_servo_angle())
    print("Current joint (rad):", arm.get_servo_angle(is_radian=True))

    # arm.move_gohome(wait=True)
    arm.disconnect()
    print("Reset joints finished.")

if __name__ == "__main__":
    main()