## 1. Installation
```
conda create -n rebarpolicy python=3.9
conda activate rebarpolicy
# install ufactory API, Install from source code
# git clone https://github.com/xArm-Developer/xArm-Python-SDK.git   # we already have the package here
cd xArm-Python-SDK
pip install .
```

### 2. Meta quest 2

### Installation

Headset should be plugged into PC, and the permissions prompt "USB debugging" should be accepted in the system of headset.

Also, the controller has to be in view of the headset cameras. It detects the pose of the handle via infrared stickers on the handle.

Teleoperation works by applying the changes to the oculus handle’s pose to the robot gripper’s pose.

### Core code
["meta_quest2.py"](io_devices/meta_quest2.py)


### Operation instruction:
* Move: **Move controller**
* Grasp: **Right trigger**
* Over: **A**

## 3. Config setting

...

## 4. Collecting demos
```
python scripts/data_collection_metaquest.py
```

# 
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot
