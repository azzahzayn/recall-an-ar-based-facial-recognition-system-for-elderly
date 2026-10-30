import os
import time
import cv2
import numpy as np
from typing import Optional
from common.schemas import FramePacket
from modules.capture.device_interface import Camera
from modules.capture.config import CameraConfig

class FileReplayCamera(Camera):
    def __init__(self, data_dir: str, config: Optional[CameraConfig] = None, loop: bool = True):
        self.data_dir = data_dir
        self.config = config or CameraConfig()
        self.loop = loop
        self._rgb_files = []
        self._depth_files = []
        self._current_index = 0
        self._started = False

    def start(self) -> None:
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Replay data directory '{self.data_dir}' does not exist.")
            
        rgb_dir = os.path.join(self.data_dir, "rgb")
        depth_dir = os.path.join(self.data_dir, "depth")
        
        if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
            # Fallback to listing matched prefixes in target directory
            all_files = sorted(os.listdir(self.data_dir))
            rgbs = [f for f in all_files if f.startswith("rgb_")]
            depths = [f for f in all_files if f.startswith("depth_")]
            self._rgb_files = [os.path.join(self.data_dir, f) for f in rgbs]
            self._depth_files = [os.path.join(self.data_dir, f) for f in depths]
        else:
            rgbs = sorted(os.listdir(rgb_dir))
            depths = sorted(os.listdir(depth_dir))
            self._rgb_files = [os.path.join(rgb_dir, f) for f in rgbs]
            self._depth_files = [os.path.join(depth_dir, f) for f in depths]
            
        if len(self._rgb_files) == 0:
            raise RuntimeError(f"No RGB frames found in '{self.data_dir}'")
        if len(self._rgb_files) != len(self._depth_files):
            raise RuntimeError(f"Mismatch: found {len(self._rgb_files)} RGB files and {len(self._depth_files)} depth files")
            
        self._current_index = 0
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> FramePacket:
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")
            
        if self._current_index >= len(self._rgb_files):
            if self.loop:
                self._current_index = 0
            else:
                raise EOFError("No more frames in replay directory.")
                
        rgb_path = self._rgb_files[self._current_index]
        depth_path = self._depth_files[self._current_index]
        
        # Load RGB
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Failed to read image at {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load Depth
        if depth_path.endswith(".npy"):
            depth = np.load(depth_path)
        else:
            # 16-bit depth png (usually in mm, divide by 1000 to get meters)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise ValueError(f"Failed to read depth image at {depth_path}")
            depth = depth_img.astype(np.float32) / 1000.0
            
        # Optional resize to match config targets
        if rgb.shape[1] != self.config.width or rgb.shape[0] != self.config.height:
            rgb = cv2.resize(rgb, (self.config.width, self.config.height))
        if depth.shape[1] != self.config.width or depth.shape[0] != self.config.height:
            depth = cv2.resize(depth, (self.config.width, self.config.height), interpolation=cv2.INTER_NEAREST)
            
        timestamp = time.time()
        self._current_index += 1
        
        return FramePacket(
            rgb=rgb,
            depth=depth,
            intrinsics=self.config.to_intrinsics(),
            timestamp=timestamp
        )
