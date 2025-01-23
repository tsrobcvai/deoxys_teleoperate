import cv2
import numpy as np
from easydict import EasyDict

from deoxys_vision.threading.threading_utils import Worker


class GoProCameraWorker(Worker):
    def __init__(
            self,
            camera_config: EasyDict = {},
            camera_address: str = '/dev/video6',
            thread_safe: bool = True,
    ):
        # import pdb;pdb.set_trace()
        self.capture = cv2.VideoCapture(camera_address) # hard-coded camera path   v4l2-ctl --list-devices
        if not self.capture.isOpened():
            raise Exception("Could not open device, you must reset the address of the camera here")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.img_w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.img_h)
        self.capture.set(cv2.CAP_PROP_FPS, camera_config.fps)

        self.last_obs = None
        self.camera_config = camera_config

        super().__init__()

    def run(self) -> None:
        self.last_obs = EasyDict()

        while not self._halt:
            ret, frame = self.capture.read()
            if not ret:
                continue
            if self.camera_config.enable_color:
                self.last_obs["color"] = frame

        self.capture.release()

    def save_img(self, img_name):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite(img_name, frame)


class GoProInterface:
    """
    This is the Python Interface for getting images from a GoPro or standard USB camera.
    """

    def __init__(
            self,
            device_id=0,
            color_cfg: dict = None,
            camera_address: str = None
    ):

        if color_cfg is not None:
            self.color_cfg = color_cfg
        else:
            self.color_cfg = EasyDict(
                enabled=True, img_w=640, img_h=480, img_format='bgr', fps=30
            )

        if not self.color_cfg.enabled:
            raise ValueError("Color stream must be enabled for GoProInterface.")

        camera_config = EasyDict(
            enable_color=self.color_cfg.enabled,
            img_w=self.color_cfg.img_w,
            img_h=self.color_cfg.img_h,
            fps=self.color_cfg.fps,
            device_id=device_id,
        )
        self.camera = GoProCameraWorker(
            camera_config=camera_config,
            thread_safe=False,
            camera_address=camera_address,
        )

    def start(self):
        self.camera.start()

    def get_last_obs(self):
        """
        Get last observation from camera
        """
        if self.camera.last_obs is None or self.camera.last_obs == {}:
            return None
        else:
            self.last_obs = self.camera.last_obs
            return self.last_obs

    def close(self):
        self.camera.halt()

    def save_image(self, img_name):
        self.camera.save_img(img_name)