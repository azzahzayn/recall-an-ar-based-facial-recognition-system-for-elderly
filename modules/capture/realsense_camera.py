import numpy as np
import time
from typing import Optional
from common.schemas import FramePacket
from modules.capture.device_interface import Camera
from modules.capture.config import CameraConfig

class RealSenseCamera(Camera):
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.pipeline = None
        self.align = None
        self._started = False
        
    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 is not installed. Please install it or use FileReplayCamera / MockCamera instead."
            )
            
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        
        # Enable color and depth streams
        rs_config.enable_stream(
            rs.stream.color, 
            self.config.width, 
            self.config.height, 
            rs.format.rgb8, 
            self.config.fps
        )
        rs_config.enable_stream(
            rs.stream.depth, 
            self.config.width, 
            self.config.height, 
            rs.format.z16, 
            self.config.fps
        )
        
        # Start pipeline
        profile = self.pipeline.start(rs_config)
        
        # Get actual intrinsics from the color stream to keep projection accurate
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        
        self.config.fx = intrinsics.fx
        self.config.fy = intrinsics.fy
        self.config.cx = intrinsics.ppx
        self.config.cy = intrinsics.ppy
        self.config.width = intrinsics.width
        self.config.height = intrinsics.height
        
        # Create alignment helper (maps depth coordinates onto color coordinates)
        self.align = rs.align(rs.stream.color)
        self._started = True

    def stop(self) -> None:
        if self._started and self.pipeline:
            self.pipeline.stop()
            self._started = False

    def read(self) -> FramePacket:
        if not self._started:
            raise RuntimeError("RealSense Camera not started. Call start() first.")
            
        import pyrealsense2 as rs
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire aligned RGB-D frame pair from RealSense.")
            
        # Convert depth and color to numpy arrays
        depth_units = depth_frame.get_units()
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * depth_units
        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        
        return FramePacket(
            rgb=color_image,
            depth=depth_image,
            intrinsics=self.config.to_intrinsics(),
            timestamp=time.time()
        )
