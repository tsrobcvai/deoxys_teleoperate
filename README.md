## 1. Installation
```
conda create -n ufact python=3.9
conda activate ufact
# install ufactory API, Install from source code
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
pip install .
pip install -r requirements.txt
cd ..
cd io_devices/oculus_reader
pip install -r requirements.txt
cd ../..
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

### 4.1 Start collecting

For Ufactory XArm6 (default):
```
python scripts/data_collection_metaquest.py
```

For Ufactory lite6:
```
python scripts/data_collection_metaquest.py --robot lite6 --ip 192.168.1.193
```

### 4.2 Reset robot pose if need
```
 python scripts/reset_robot_joints.py 
 
 python scripts/reset_robot_cartesian.py
```


# 
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot

### debug