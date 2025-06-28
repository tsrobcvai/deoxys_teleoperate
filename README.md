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
```
python scripts/data_collection_metaquest.py
```

# 
Our repo was built on: 
Deoxys, 
Deoxys vision, 
GRoot

### debug
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.06631600000003, -112.312558, 251.56450300000003, 2.998765025315816, 0.19586484852830702, 1.6493232301833172], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.054316, -112.290358, 251.54850300000004, 2.998837422242626, 0.19594962142092245, 1.64883107465501], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.044316, -112.33455799999999, 251.52650300000005, 2.9987405663467293, 0.196251851775199, 1.648144742738038], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.058316, -112.29855799999999, 251.55750300000005, 2.998650690225396, 0.19681757427301672, 1.6473023146523287], velo=20.0, acc=200.0, is_tool_coord=False
ControllerError, code: 24
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.041316, -112.296258, 251.57450300000005, 2.997970505829112, 0.19719814431313146, 1.6467370562347816], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.966316, -112.30075799999999, 251.59950300000003, 2.9970578299308905, 0.19809781828473327, 1.6460961385775348], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.78531599999997, -112.461058, 251.57950300000005, 2.992723991328819, 0.19908867577152356, 1.6466660596494076], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.86231599999996, -112.41455799999999, 251.71950300000003, 2.9924184917179693, 0.20277041723853942, 1.644951506171325], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.80431600000003, -112.41775799999999, 251.73750300000003, 2.992236490409293, 0.20351696381368714, 1.6451593402056572], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.896316, -112.466258, 251.76450300000002, 2.9921599364552574, 0.20364759215609918, 1.646165394724052], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.938316, -112.32215799999999, 251.74950300000003, 2.9933588382664578, 0.2061101307715849, 1.6466456955703945], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[391.945316, -111.950858, 251.42150300000006, 2.996861381435613, 0.2192003971263501, 1.6489290081879713], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.054316, -111.88295799999999, 251.08850300000006, 2.9966736269506535, 0.22295736494588606, 1.6492616822149244], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.143316, -111.53495799999999, 251.00950300000002, 3.0010513103993333, 0.22604035840518422, 1.6474390788887534], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> set_servo_cartisian -> code=1, pose=[392.108316, -111.576358, 251.09150300000002, 2.999234208285446, 0.2216977107987408, 1.6475222744083295], velo=20.0, acc=200.0, is_tool_coord=False
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> ft_sensor_app_set -> code=1, app_code=0
[SDK][ERROR][2025-06-27 15:37:03][base.py:381] - - API -> ft_sensor_enable -> code=1, on_off=0