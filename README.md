## 1. Installation
```
conda create -n rebarpolicy python=3.9
conda activate rebarpolicy
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
```
python data_collection_metaquest.py
```

# 
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot

### debug

Traceback (most recent call last):
  File "/home/zekaijin/ufactory/deoxys_teleoperate/data_collection_metaquest.py", line 5, in <module>
    from io_devices import Meta_quest2
  File "/home/zekaijin/ufactory/deoxys_teleoperate/io_devices/__init__.py", line 2, in <module>
    from .meta_quest2 import Meta_quest2
  File "/home/zekaijin/ufactory/deoxys_teleoperate/io_devices/meta_quest2.py", line 5, in <module>
    from oculus_reader.oculus_reader.reader import OculusReader
ModuleNotFoundError: No module named 'oculus_reader