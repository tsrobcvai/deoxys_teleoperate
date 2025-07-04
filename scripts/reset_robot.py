import time
import numpy as np
from xarm.wrapper import XArmAPI


def main(robot: str = "xarm6",
         ip: str | None = None,
         pose: list[float] | None = None):

    pose = np.array(pose, dtype=np.float32)

    print(f"Connecting to {robot} at {ip} ...")
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.clean_error()

    # manual mode first (mode 2 = position/impedance)
    arm.set_mode(2)
    arm.set_state(0)

    # ask operator for confirmation
    resp = input("Reset the arm to this pose? (y to confirm): ").strip().lower()
    if resp != "y":
        print("Aborted by user.")
        arm.disconnect()
        return

    # switch to position control (mode 0) and move
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"Moving to pose: {pose} (units: mm, deg)")
    arm.set_position(*pose, wait=True) 

    arm.disconnect()
    print("Reset finished.")

# ==========================================================================
# Configure your robot here
# We will get in to the manual mode first, after the user confirms, we will switch to position control mode
# ==========================================================================
if __name__ == "__main__":
    ROBOT = "xarm6"                 # "xarm6" | "lite6" | "uf850"
    IP    = None                    # None → use default for ROBOT
    POSE  = None                    # None → use robot’s default pose

    # POSE example: [300, 50, 350, -180, 0, 0]

    main(robot=ROBOT, ip=IP, pose=POSE)
