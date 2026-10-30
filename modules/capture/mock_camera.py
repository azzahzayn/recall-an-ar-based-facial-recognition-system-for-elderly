import numpy as np
import time
from typing import Optional
from common.schemas import FramePacket
from modules.capture.device_interface import Camera
from modules.capture.config import CameraConfig

class MockCamera(Camera):
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> FramePacket:
        if not self._started:
            raise RuntimeError("MockCamera not started. Call start() first.")
            
        # Create a synthetic image with a simple face-like shape
        rgb = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        # Background: dark blue
        rgb[:, :, 0] = 20
        rgb[:, :, 2] = 40
        
        # Face-like oval in the center
        cy, cx = self.config.height // 2, self.config.width // 2
        ry, rx = int(self.config.height * 0.3), int(self.config.width * 0.2)
        
        y, x = np.ogrid[:self.config.height, :self.config.width]
        mask = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0
        
        # Color the oval (skin tone-ish)
        rgb[mask] = [220, 180, 140]
        
        # Draw eyes and mouth inside the oval
        # Left eye
        le_y, le_x = cy - ry // 3, cx - rx // 3
        rgb[(y - le_y)**2 + (x - le_x)**2 <= 64] = [20, 20, 20]
        # Right eye
        re_y, re_x = cy - ry // 3, cx + rx // 3
        rgb[(y - re_y)**2 + (x - re_x)**2 <= 64] = [20, 20, 20]
        # Mouth
        m_y, m_x = cy + ry // 3, cx
        m_mask = ((y - m_y)**2 + (x - m_x)**2 <= 144) & (y >= m_y)
        rgb[m_mask] = [150, 40, 40]
        
        # Create a synthetic depth map (sphere-like in the face region)
        depth = np.ones((self.config.height, self.config.width), dtype=np.float32) * 2.0  # 2m background
        
        # Face is closer, e.g. a hemisphere shape
        dist_sq = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
        face_depth_mask = dist_sq <= 1.0
        # Smooth depth from 1.0m (center) to 1.3m (edges)
        depth[face_depth_mask] = 1.0 + 0.3 * dist_sq[face_depth_mask]
        
        # Add slight noise to depth
        noise = np.random.normal(0, 0.005, depth.shape).astype(np.float32)
        depth += noise
        
        return FramePacket(
            rgb=rgb,
            depth=depth,
            intrinsics=self.config.to_intrinsics(),
            timestamp=time.time()
        )
