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
    parser.add_argument('--joint', type=float, nargs=6, default=None,
                        help='Target joint positions (deg or rad)')
    parser.add_argument('--is-radian', action='store_true', help='Interpret joint values as radians')
    parser.add_argument('--speed', type=float, default=50, help='Joint move speed (deg/s or rad/s)')
    return parser.parse_args()

def get_default_joints(robot_type):
    """Get default joint angles based on robot type"""
    if robot_type == "lite6":
        return [0, 0, 90, 0, 90 , -17]  
    elif robot_type == "xarm6":
        return [5.4, 0.9, -92.6, 1.1, 90, 2.2]  
    elif robot_type == "uf850":
        return [0, 0, 60, 0, 60, -17]
    else:
        return [0, 0, 60, 0, 60, -17]  

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

    if args.joint is None:
        joint = np.array(get_default_joints(args.robot), dtype=np.float32)
    else:
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

    arm.disconnect()
    print("Reset joints finished.")

if __name__ == "__main__":
    main()