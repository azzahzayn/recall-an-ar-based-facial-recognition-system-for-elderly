# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import time
from typing import Optional
from common.schemas import FramePacket
from modules.capture.device_interface import Camera
from modules.capture.config import CameraConfig

class WebcamCamera(Camera):
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap = None
        self._started = False

    def start(self) -> None:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Ensure a camera is connected and not in use by another app.")
        
        # Try to configure the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Read actual resolution set by hardware
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w > 0 and actual_h > 0:
            self.config.width = actual_w
            self.config.height = actual_h
            # Recalculate cx, cy dynamically
            self.config.cx = actual_w / 2.0
            self.config.cy = actual_h / 2.0
            
        self._started = True

    def stop(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        self._started = False

    def read(self) -> FramePacket:
        if not self._started or not self.cap:
            raise RuntimeError("WebcamCamera not started. Call start() first.")
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam.")
            
        # OpenCV captures in BGR, convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a dummy depth map (webcam does not have depth)
        depth = np.ones((self.config.height, self.config.width), dtype=np.float32)
        
        return FramePacket(
            rgb=rgb,
            depth=depth,
            intrinsics=self.config.to_intrinsics(),
            timestamp=time.time()
        )
