import numpy as np
import pytest
from common.schemas import Landmarks5pt, BBox
from modules.detection.alignment import align_face
from modules.detection.detector import FaceDetector

def test_align_face():
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Define custom landmarks matching normal face bounds
    landmarks = Landmarks5pt(
        points=np.array([
            [30, 45], # left eye
            [70, 45], # right eye
            [50, 60], # nose tip
            [35, 80], # mouth left
            [65, 80]  # mouth right
        ], dtype=np.float32)
    )
    
    # Execute face alignment transformation
    warped = align_face(img, landmarks, target_size=(112, 112))
    assert warped.shape == (112, 112, 3)

def test_detector_empty_frame():
    detector = FaceDetector()
    
    # Create an empty black image
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Assert detection gracefully returns empty lists rather than crashing
    res = detector.detect(img)
    assert isinstance(res.boxes, list)
    assert isinstance(res.landmarks, list)
    assert len(res.boxes) == 0
