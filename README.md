## 1. Installation
```
conda create -n rebarpolicy python=3.9
conda activate rebarpolicy
```
### 1.1 Deoxys
Install System Prerequisite of Deoxys: https://zhuyifengzju.github.io/deoxys_docs/html/installation/system_prerequisite.html
We already have the repo here
```
cd deoxys_control/deoxys
./InstallPackage # choose 0.13.3 if you are in >= 5.5.0 (FR3)
make -j build_deoxys=1
pip install -U -r requirements.txt
```
### 1.2 Deoxys vision
```
cd ../../
cd deoxys_vision
pip install -e .
```
## 2. Meta quest 2
### Installation
Headset should be plugged into PC, and the permissions prompt "USB debugging" should be accepted in the system of headset.
Also, the controller has to be in view of the headset cameras. It detects the pose of the handle via infrared stickers on the handle.
Teleoperation works by applying the changes to the oculus handle’s pose to the robot gripper’s pose.
### Core code
["meta_quest2.py"](deoxys_control/deoxys/deoxys/utils/io_devices/meta_quest2.py)
### Operation instruction:
* Move: **Move controller**
* Grasp: **Right trigger**
* Over: **A**
## 3. Config setting
Please modify ["real_robot_observation_cfg.yml"](rebar_configs/real_robot_observation_cfg.yml)
## 4. Collecting demos

### 4.1 Reset robot pose if need
```
export PYTHONPATH=$PYTHONPATH:./deoxys_control/deoxys
python rebar_scripts/reset_robot_joints.
```
### 4.2 Open cameras needed, such as
```
python deoxys_vision/scripts/deoxys_camera_node.py --camera-ref gopro_0 --use-rgb --visualization --img-h 720 --img-w 1280 --fps 30 --camera-address '/dev/video6'
python deoxys_vision/scripts/deoxys_camera_node.py --camera-ref webcam_2 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --camera-address '/dev/video2'
python deoxys_vision/scripts/deoxys_camera_node.py --camera-ref rs_1 --use-rgb --use-depth --visualization --img-h 480 --img-w 640 --fps 30
```
if you want to check the cameras plugged
```
v4l2-ctl --list-devices
```

### 4.2 Start collecting
```
export PYTHONPATH=$PYTHONPATH:./deoxys_control/deoxys
python rebar_scripts/deoxys_data_collection_quest2_abs.py
```
Note: After pressing "A", we will be prompted to choose if save the demo; Then, we have to manully close the process since it won't shut down automatically (a bug need to be fixed)

# 
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot
