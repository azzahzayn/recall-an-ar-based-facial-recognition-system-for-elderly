import os
import pytest
import numpy as np
import time
from common.schemas import BBox, FramePacket, CameraIntrinsics, Embedding
from modules.recognition.config import RecognitionConfig
from modules.recognition.identity_tracker import IdentityTracker, compute_iou
from modules.recognition.pipeline import RecognitionPipeline
from modules.detection.detector import FaceDetector
from modules.embedding.config import EmbedderConfig
from modules.embedding.embedder import FaceEmbedder
from modules.database.repository import IdentityRepository
from modules.database.config import DatabaseConfig

def test_compute_iou():
    box1 = BBox(xmin=10, ymin=10, xmax=50, ymax=50, score=1.0)
    box2 = BBox(xmin=20, ymin=20, xmax=60, ymax=60, score=1.0)
    
    # Intersection: [20, 50] x [20, 50] = 30 * 30 = 900
    # Box1 Area: 40 * 40 = 1600
    # Box2 Area: 40 * 40 = 1600
    # Union Area: 1600 + 1600 - 900 = 2300
    # IoU: 900 / 2300 = 0.3913
    assert pytest.approx(compute_iou(box1, box2), 1e-4) == 900.0 / 2300.0

def test_tracker_smoothing():
    config = RecognitionConfig(max_track_history=5, track_iou_threshold=0.3, track_max_age=1.0)
    tracker = IdentityTracker(config)
    
    box1 = BBox(xmin=10, ymin=10, xmax=50, ymax=50, score=1.0)
    box2 = BBox(xmin=12, ymin=12, xmax=52, ymax=52, score=1.0)
    
    t = time.time()
    # Frame 1: Initial match
    res1 = tracker.update([box1], ["Alice"], [0.9], t)
    assert len(res1) == 1
    assert res1[0].identity == "Alice"
    
    # Frame 2: Slight movement, transient database mismatch (identity is None)
    res2 = tracker.update([box2], [None], [0.0], t + 0.1)
    assert len(res2) == 1
    # Asserts that smoothing resolves transient identity failures to keep continuous AR label active
    assert res2[0].identity == "Alice"
    
    # Frame 3: Age out track with empty update
    res3 = tracker.update([], [], [], t + 2.0)
    assert len(res3) == 0
    assert len(tracker.tracks) == 0

def test_recognition_pipeline(tmp_path):
    db_file = os.path.join(tmp_path, "rec_test.db")
    db_cfg = DatabaseConfig(db_path=db_file, similarity_threshold=0.5)
    repo = IdentityRepository(db_cfg)
    
    detector = FaceDetector()
    embedder = FaceEmbedder(EmbedderConfig(use_mock=True))
    pipeline = RecognitionPipeline(detector, embedder, repo)
    
    # Create fake RGB frame
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[80:160, 100:180] = [220, 180, 140]  # Skin tone square
    depth = np.ones((240, 320), dtype=np.float32) * 1.0
    
    intrinsics = CameraIntrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)
    packet = FramePacket(rgb=rgb, depth=depth, intrinsics=intrinsics, timestamp=time.time())
    
    # Execute pipeline process
    results = pipeline.process_frame(packet)
    assert isinstance(results, list)
    
    repo.close()
