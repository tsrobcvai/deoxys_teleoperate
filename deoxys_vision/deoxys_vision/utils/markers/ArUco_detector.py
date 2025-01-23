import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class ArUcoDetector:
    def __init__(self, dictionary=cv2.aruco.DICT_6X6_250):
        """
        Initialize the ArUco detector with a specified dictionary.
        Default dictionary: DICT_6X6_250.
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv2.aruco.DetectorParameters()
        self.results = []

    def detect(self, img: np.ndarray, img_idx: int, intrinsics: dict, tag_size: float):
        """
        Detect ArUco markers in the input image and estimate their pose.

        Args:
        - img: Input image as a numpy array.
        - img_idx: Index of the image for saving detected visualization.
        - intrinsics: Dictionary containing camera intrinsic parameters.
        - marker_length: The length of the marker's side (in meters or any consistent unit).

        Returns:
        - self.results: List of detected marker information.
        """
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )
        self.results = []
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                tag_size,
                cameraMatrix=np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                                       [0, intrinsics["fy"], intrinsics["cy"]],
                                       [0, 0, 1]]),
                distCoeffs=None,
            )

            self.results = [{"id": int(id[0]), "pose_R": R.from_rotvec(rvec).as_matrix()[0], "pose_t": tvec}
                            for id, rvec, tvec in zip(ids, rvecs, tvecs)]

            detected_img = self.vis_markers(img, corners, ids, rvecs, tvecs, intrinsics)

            cv2.imwrite(f'rebar_scripts/pose_estimation/eye2hand_base/calibration_imgs/detected_{img_idx}.png',
                        detected_img)
        # else:
        #     print(f"wrong detection, skipping detection of image {img_idx}")

        return self.results

    def __len__(self):
        return len(self.results)

    def vis_markers(self, img: np.ndarray, corners, ids, rvecs, tvecs, intrinsics):
        """
        Visualize detected ArUco markers with their bounding boxes and IDs.
        """
        img = img.astype(np.uint8)
        if ids is not None:
            for i, corner in enumerate(corners):
                points = corner[0].astype(int)
                ptA, ptB, ptC, ptD = points
                # Draw the bounding box
                cv2.line(img, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
                cv2.line(img, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
                cv2.line(img, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
                cv2.line(img, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

                # Draw the center
                cX, cY = np.mean(points, axis=0).astype(int)
                cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

                # Draw the ID
                tag_id = int(ids[i][0])
                cv2.putText(
                    img,
                    f"ID-{tag_id}",
                    (ptA[0], ptA[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Draw the pose axes if rvecs and tvecs are available
                if rvecs is not None and tvecs is not None:
                    cv2.drawFrameAxes(
                        img,
                        np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                                  [0, intrinsics["fy"], intrinsics["cy"]],
                                  [0, 0, 1]]),
                        None,
                        rvecs[i],
                        tvecs[i],
                        0.05
                    )

        return img

    def tags_centroid(self):
        """
        Return a dictionary with tag IDs as keys and centroids as values.
        """
        centroid_dict = {}
        for result in self.results:
            tag_id = result["id"]
            tvec = result["tvec"][0]  # Translation vector
            centroid_dict[tag_id] = tvec.tolist()  # Store as a list
        return centroid_dict
