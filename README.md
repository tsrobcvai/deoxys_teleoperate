# UFACTORY TELEOPERATION

Experiments for Ufactory Xarm Robots

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

This repo supports Meta Quest 2 as the teleoperation device. To start, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader/blob/main/oculus_reader/reader.py) by following instructions in the link.

["meta_quest2.py"](io_devices/meta_quest2.py)


### Operation instruction:
* Move: **Move controller**
* Grasp: **Right trigger**
* Over: **A**

## 3. Config setting

...

## 4. Collecting demos


Example usage (note that `camera_node.py` and `data_collection_metaquest.py` should be run simultaneously in at least two separate windows):


### 4.1 Reset robot pose if need


For Ufactory XArm6 (default):

```
 python scripts/reset_robot_joints.py 
 
 python scripts/reset_robot_cartesian.py
```

For other robots (i.e. Ufactory lite6)

```
python scripts/reset_robot_joints.py  --robot lite6 --ip 192.168.1.193

python scripts/reset_robot_cartesian.py  --robot lite6 --ip 192.168.1.193
```

### 4.2 Open cameras needed, such as

```
python scripts/camera_node.py --camera-ref gopro_0 --use-rgb --visualization --img-h 720 --img-w 1280 --fps 30 --camera-address '/dev/video6'

python scripts/camera_node.py --camera-ref webcam_2 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --camera-address '/dev/video2'

python scripts/camera_node.py --camera-ref rs_1 --use-rgb --use-depth --visualization --img-h 480 --img-w 640 --fps 15
```

if you want to check the cameras plugged,

```
v4l2-ctl --list-devices
```

Node processes can be cleaned up by running `pkill -9 -f scripts/camera_node.py`.

### 4.3 Start collecting

For Ufactory XArm6 (default):

```
python scripts/data_collection_metaquest.py
```

For other robots (i.e. Ufactory lite6)

```
python scripts/data_collection_metaquest.py --robot lite6 --ip 192.168.1.193
```

# Acknowledgement
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot

### debug