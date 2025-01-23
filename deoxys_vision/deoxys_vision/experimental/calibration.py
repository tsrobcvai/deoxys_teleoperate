import os
import json
import cv2

import numpy as np

from deoxys_vision import ROOT_PATH
from deoxys_vision.utils.markers.apriltag_detector import AprilTagDetector
# from deoxys_vision.utils.markers.ArUco_detector import ArUcoDetector
from deoxys_vision.utils.transformation.transform_manager import TransformManager
from deoxys_vision.utils.robot_urdf_models.urdf_models import PandaURDFModel
import deoxys_vision.utils.transformation.transform_utils as T
from scipy.spatial.transform import Rotation as R

class HandEyeCalibrationBase():
    def __init__(self,
                 asset_folder=os.path.join(ROOT_PATH, "utils/robot_urdf_models"),
                 assistive_marker_type="apriltag", # org "apriltag"   "aruco"
                 robot="Panda"):
        assert(assistive_marker_type in ["apriltag", "aruco"])
        assert(robot in ["Panda"])

        self._assistive_marker_type = assistive_marker_type
        if self._assistive_marker_type == "apriltag":
            self._marker_detector = AprilTagDetector()
            self._marker_configuration = {"intrinsics": None, "tag_size": None}
        elif self._assistive_marker_type == "aruco":
            self._marker_detector = ArUcoDetector()
            self._marker_configuration = {"intrinsics": None, "tag_size": None}
        else:
            raise NotImplementedError
        self._transform_manager = TransformManager()

        self._robot_model = PandaURDFModel(asset_folder=asset_folder)

        self.rot_target2cam_list = []
        self.pos_target2cam_list = []

        self.counter = 0

    def reset(self):
        self.rot_target2cam_list = []
        self.pos_target2cam_list = []
        self.counter = 0

    def update_detector_configs(self, **kwargs):
        self._marker_configuration.update(kwargs)

    def detect_marker(self, rgb_img, img_idx):
        """
        Args:
           rgb_img (np.array): HxWx3 RGB images for detecting markers
        Return:
           detected pose of the marker, in a tuple (R, t)
        """
        # try:
        detect_results = self._marker_detector.detect(rgb_img, img_idx, **self._marker_configuration)
        # except:
        #     print("1")

        if len(detect_results) != 1:
            print(f"wrong detection, skipping detection of image {img_idx}")
            return None
        else:
            # if marker type is April
            rot = np.array(detect_results[0].pose_R)
            pos = np.array(detect_results[0].pose_t)

            # if marker type is ArUco
            # rot = np.array(detect_results[0]["pose_R"])
            # pos = np.array(detect_results[0]["pose_t"])
            return (pos, rot)

    def get_gripper_pose_from_model(self, joint_configuration):
        pose = self._robot_model.get_gripper_pose(joint_configuration)[:2]
        pos = np.array(pose[0])
        rot = np.array(T.quat2mat(pose[1]))
        return (pos, rot)

    def get_marker_pose_from_model(self, joint_configuration):
        """
        The robot model has the model of marker, which can be used to compute the pose of the marker based on the robot model.
        """
        pose = self._robot_model.get_marker_pose(joint_configuration)[:2]
        pos = np.array(pose[0])
        rot = np.array(T.quat2mat(pose[1]))
        return (pos, rot)
    
    def add_transform(self, target_frame_name, source_frame_name, R, pos):
        """
        Store the transformation of `target_frame` relative to `source_frame`
        """
        self._transform_manager.add_transform(target_frame_name, source_frame_name, R, pos)

    def get_transform(self, target_frame_name, source_frame_name):
        """
        Store the transformation of `target_frame` relative to `source_frame`
        """
        self._transform_manager.get_transform(target_frame_name, source_frame_name)

    def step(self, rgb_img, robot_joints, verbose=False):
        """
        Args:
           rgb_img (np.array): 
           robot_joints (np.array): 
        """
        raise NotImplementedError

    def _calibrate_function(self,
                            pos_1_list, # self.pos_gripper2base_list,
                            rot_1_list, # self.rot_gripper2base_list,
                            pos_2_list, # self.pos_target2cam_list,
                            rot_2_list, # self.rot_target2cam_list,
                            calibration_method="tsai",
                            verbose=True):
        method = None
        if calibration_method == "tsai":
            method = cv2.CALIB_HAND_EYE_TSAI
        elif calibration_method == "horaud":
            method = cv2.CALIB_HAND_EYE_HORAUD

        rot, pos = cv2.calibrateHandEye(
            rot_1_list,
            pos_1_list,
            rot_2_list,
            pos_2_list,
            method=method
        )

        # avg_trans_error, avg_rot_error = self.calculate_calibration_error(
        #     rot, pos, rot_1_list, pos_1_list, rot_2_list, pos_2_list
        # )
        # print("avg_trans_error: ", avg_trans_error)
        # print("avg_rot_error: ", avg_rot_error)

        if verbose:
            # print("Rotation matrix: ", rot)
            print("Axis Angle: ", T.quat2axisangle(T.mat2quat(rot)))
            print("Rotation Mat: ", rot)
            print("Quaternion: ", T.mat2quat(rot))
            print("Translation: ", pos.transpose())
        return (pos, rot)

    # Tao
    # def calculate_calibration_error(self, rot, pos, rot_1_list, pos_1_list, rot_2_list, pos_2_list):
    #
    #
    #     trans_errors = []
    #     rot_errors = []
    #     # Estimated R_cam2gripper, t_cam2gripper
    #     T_ce = np.eye(4)
    #     T_ce[:3, :3] = rot
    #     T_ce[:3, 3] = pos.flatten()
    #
    #     for i in range(len(rot_1_list)):
    #         # Actual ee to base transformation (T_e_b)
    #         T_eb = np.eye(4)
    #         T_eb[:3, :3] = rot_1_list[i]
    #         T_eb[:3, 3] = pos_1_list[i].flatten()
    #
    #         # Actual target to camera transformation (T_c_b_actual)                Actual camera to base transformation (T_c_b_actual)
    #         T_cb_actual = np.eye(4)
    #         T_cb_actual[:3, :3] = rot_2_list[i]
    #         T_cb_actual[:3, 3] = pos_2_list[i].flatten()
    #
    #
    #         raise NotImplementedError
    #
    #         # TODO: below is incorrect
    #         # Predicted camera to base transformation (T_c_b_pred)
    #         T_cb_pred = T_ce @ T_eb
    #
    #         # Translation error
    #         trans_error = np.linalg.norm(T_cb_actual[:3, 3] - T_cb_pred[:3, 3])
    #         trans_errors.append(trans_error)
    #         # Rotation error
    #         R_actual = T_cb_actual[:3, :3]
    #         R_pred = T_cb_pred[:3, :3]
    #         R_diff = R.from_matrix(R_actual @ R_pred.T)
    #         rot_error = R_diff.magnitude() * (180 / np.pi)  # Convert radian to degree
    #         rot_errors.append(rot_error)
    #     # Average translation and rotation error
    #     # print(trans_errors)
    #     avg_trans_error = np.mean(trans_errors)
    #     # print(rot_errors)
    #     avg_rot_error = np.mean(rot_errors)
    #     return avg_trans_error, avg_rot_error


    def calibrate(self,
                  calibration_method="tsai",
                  verbose=True):
        raise NotImplementedError

    def save_calibration(self, pos, rot, config_folder, json_file_name):
        with open(
            os.path.join(
                config_folder,
                json_file_name,
                # f"camera_{args.camera_id}_{args.camera_type}_{args.post_fix}extrinsics.json",
            ),
            "w",
        ) as f:
            extrinsics = {"translation": pos.tolist(), "rotation": rot.tolist()}
            json.dump(extrinsics, f)
        

class EyeInHandCalibration(HandEyeCalibrationBase):
    def __ini__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.pos_gripper2base_list = []
        self.pos_gripper2base_list = []

    def reset(self):
        super().reset()
        self.rot_gripper2base_list = []
        self.pos_gripper2base_list = []

    def step(self, rgb_img, robot_joints, img_idx, verbose=False):
        result = self.detect_marker(rgb_img, img_idx)
        if result is None:
            if verbose:
                print(f"Detection result does not comply with calibration method, discarding this observation")
            return None
        (marker_pos_in_cam, marker_rot_in_cam) = result
        if verbose:
            print(f"Detection result: {result}")

        (gripper_pos_in_base, gripper_rot_in_base) = self.get_gripper_pose_from_model(robot_joints)

        self.add_transform(f"target_{self.counter}", "cam_view_{self.counter}", gripper_rot_in_base, gripper_pos_in_base)        
        self.add_transform(f"ee_{self.counter}", "base", gripper_rot_in_base, gripper_pos_in_base)
        self.pos_target2cam_list.append(marker_pos_in_cam)
        self.rot_target2cam_list.append(marker_rot_in_cam)

        self.pos_gripper2base_list.append(gripper_pos_in_base[..., np.newaxis])
        self.rot_gripper2base_list.append(gripper_rot_in_base)
        self.counter += 1

    def calibrate(self,
                  calibration_method="tsai",
                  verbose=True):
        (cam_pos_in_gripper, cam_rot_in_gripper) = self._calibrate_function(self.pos_gripper2base_list,
                                                                      self.rot_gripper2base_list,
                                                                      self.pos_target2cam_list,
                                                                      self.rot_target2cam_list,
                                                                      calibration_method=calibration_method,
                                                                      verbose=verbose)
        for i in range(self.counter):
            self.add_transform(f"cam_view_{self.counter}", f"ee_{self.counter}", cam_rot_in_gripper, cam_pos_in_gripper)
        return (cam_pos_in_gripper, cam_rot_in_gripper)

class EyeToHandCalibration(HandEyeCalibrationBase):
    def __ini__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.rot_gripper2base_list = []
        self.pos_gripper2base_list = []

    def reset(self):
        super().reset()
        self.rot_gripper2base_list = []
        self.pos_gripper2base_list = []

    def step(self, rgb_img, robot_joints, img_idx, verbose=False):
        result = self.detect_marker(rgb_img, img_idx)
        if result is None:
            return None
        (marker_pos_in_cam, marker_rot_in_cam) = result
        if verbose:
            print(f"Detection result: {result}")

        (gripper_pos_in_base, gripper_rot_in_base) = self.get_gripper_pose_from_model(robot_joints) # get ee pose using urdf model

        #TODO: why we need this?
        # self.add_transform(f"target_{self.counter}", f"cam_view_{self.counter}", gripper_rot_in_base, gripper_pos_in_base)

        self.add_transform(f"ee_{self.counter}", "base", gripper_rot_in_base, gripper_pos_in_base)
        base_transformation_in_gripper = self._transform_manager.get_transform("base", f"ee_{self.counter}")
        gripper_in_base_pos = base_transformation_in_gripper[:3, -1:]
        gripper_in_base_rot = base_transformation_in_gripper[:3, :3]

        # print(base_rot_in_gripper, gripper_rot_in_base.transpose())
        # print("=============")
        # print(base_pos_in_gripper, -gripper_rot_in_base.transpose() @ gripper_pos_in_base[:, np.newaxis])
        # print("=============")
        
        # assert(gripper_pos_in_base.shape == base_pos_in_gripper.shape)
        self.pos_target2cam_list.append(marker_pos_in_cam)
        self.rot_target2cam_list.append(marker_rot_in_cam)

        self.pos_gripper2base_list.append(gripper_in_base_pos)
        self.rot_gripper2base_list.append(gripper_in_base_rot)
        self.counter += 1

    def calibrate(self,
                  calibration_method="tsai",
                  verbose=True):
        (cam_pos_in_base, cam_rot_in_base) = self._calibrate_function(self.pos_gripper2base_list,
                                                                      self.rot_gripper2base_list,
                                                                      self.pos_target2cam_list,
                                                                      self.rot_target2cam_list,
                                                                      calibration_method=calibration_method,
                                                                      verbose=verbose)
        return (cam_pos_in_base, cam_rot_in_base)
