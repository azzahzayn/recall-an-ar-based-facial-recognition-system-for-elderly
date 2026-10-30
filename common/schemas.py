from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

@dataclass
class FramePacket:
    rgb: np.ndarray        # (H, W, 3) uint8 RGB image
    depth: np.ndarray      # (H, W) float32 depth map (in meters or mm)
    intrinsics: CameraIntrinsics
    timestamp: float       # Epoch timestamp in seconds

@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    score: float

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

@dataclass
class Landmarks5pt:
    # 5 landmarks: [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
    # Coordinates are stored as (x, y) absolute pixels
    points: np.ndarray  # (5, 2) float32 array

@dataclass
class DetectionResult:
    boxes: List[BBox] = field(default_factory=list)
    landmarks: List[Landmarks5pt] = field(default_factory=list)

@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) float32 array
    colors: np.ndarray  # (N, 3) uint8 RGB colors

@dataclass
class FaceMesh:
    vertices: np.ndarray          # (V, 3) float32 array
    faces: np.ndarray             # (F, 3) int32 array
    shape_params: Optional[np.ndarray] = None  # 3DMM shape parameters
    expression_params: Optional[np.ndarray] = None  # 3DMM expression parameters
    pose_params: Optional[np.ndarray] = None  # 3D pose parameters

@dataclass
class TexturedFaceMesh:
    mesh: FaceMesh
    texture: Optional[np.ndarray] = None  # Optional texture map image
    vertex_colors: Optional[np.ndarray] = None  # (V, 3) colors per vertex

@dataclass
class Embedding:
    vector: np.ndarray  # (512,) float32 L2-normalized vector
    model_version: str

@dataclass
class Identity:
    identity_id: str
    name: str
    embeddings: List[Embedding] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class RecognitionResult:
    bbox: BBox
    identity: Optional[str]  # None if unknown
    confidence: float        # Score in range [0, 1]
    timestamp: float
