## 1. Installation
```
conda create -n rebarpolicy python=3.9
conda activate rebarpolicy
# install ufactory API, Install from source code
# git clone https://github.com/xArm-Developer/xArm-Python-SDK.git   # we already have the package here
cd xArm-Python-SDK
pip install .

```
### 1.1 Deoxys
pip install -U -r requirements.txt
```
### 1.2 Deoxys vision
```
cd ../../
cd deoxys_vision
pip install -e .

```

### 2. Meta quest 2

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

...

## 4. Collecting demos

### 4.1 Reset robot pose if need
```
export PYTHONPATH=$PYTHONPATH:./deoxys_control/deoxys
python rebar_scripts/reset_robot_joints.
```
### 4.2 Open cameras needed, such as

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
