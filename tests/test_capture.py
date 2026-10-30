import pytest
import numpy as np
from modules.capture.config import CameraConfig
from modules.capture.mock_camera import MockCamera
from common.schemas import FramePacket

def test_mock_camera():
    config = CameraConfig(width=320, height=240, fps=15)
    cam = MockCamera(config)
    
    # Test reading before starting raises error
    with pytest.raises(RuntimeError):
        cam.read()
        
    cam.start()
    packet = cam.read()
    
    assert isinstance(packet, FramePacket)
    assert packet.rgb.shape == (240, 320, 3)
    assert packet.depth.shape == (240, 320)
    assert packet.intrinsics.width == 320
    assert packet.intrinsics.height == 240
    assert packet.intrinsics.fx == 615.0
    assert packet.intrinsics.cx == 160.0
    assert packet.intrinsics.cy == 120.0
    
    # Check that the simulated depth map has non-trivial values
    assert np.min(packet.depth) < 2.0
    assert np.max(packet.depth) <= 2.1
    
    cam.stop()
