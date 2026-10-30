import pytest
import numpy as np
from common.schemas import FramePacket, CameraIntrinsics, Landmarks5pt
from modules.pointcloud.backprojection import PointCloudProcessor
from modules.reconstruction.model_wrapper import FaceReconstructor, ParametricFaceModel
from modules.color_projection.projector import ColorProjector
from modules.pose_generation.renderer import PoseGenerator

def test_pointcloud_processor():
    processor = PointCloudProcessor()
    
    # Create simple RGB and Depth matrices
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[40:60, 40:60] = [255, 0, 0]
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[40:60, 40:60] = 1.0  # 1 meter away
    
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)
    packet = FramePacket(rgb=rgb, depth=depth, intrinsics=intrinsics, timestamp=0.0)
    
    pc = processor.back_project(packet)
    assert len(pc.points) == 400
    assert pc.points[0, 2] == 1.0  # Z coordinate should map depth values
    
    # Statistical outlier filter test
    # Inject one clear noise point far away
    pc.points = np.vstack([pc.points, [10.0, 10.0, 10.0]])
    pc.colors = np.vstack([pc.colors, [0, 0, 0]])
    
    filtered_pc = processor.remove_statistical_outliers(pc, k=5, std_ratio=1.0)
    assert len(filtered_pc.points) < len(pc.points)

def test_parametric_reconstructor():
    model = ParametricFaceModel(grid_res=15)
    assert model.vertices.shape == (225, 3)
    assert len(model.faces) > 0
    
    reconstructor = FaceReconstructor()
    intrinsics_mat = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    landmarks = Landmarks5pt(
        points=np.array([
            [40, 40],  # Left Eye
            [60, 40],  # Right Eye
            [50, 50],  # Nose Tip
            [45, 60],  # Mouth Left
            [55, 60]   # Mouth Right
        ], dtype=np.float32)
    )
    
    mesh, rvec, tvec = reconstructor.reconstruct(landmarks, intrinsics_mat)
    assert mesh.vertices.shape == (225, 3)
    assert rvec.shape == (3, 1)
    assert tvec.shape == (3, 1)

def test_color_projection_and_pose_generation():
    reconstructor = FaceReconstructor()
    intrinsics_mat = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    landmarks = Landmarks5pt(
        points=np.array([
            [40, 40],
            [60, 40],
            [50, 50],
            [45, 60],
            [55, 60]
        ], dtype=np.float32)
    )
    
    mesh, rvec, tvec = reconstructor.reconstruct(landmarks, intrinsics_mat)
    
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)
    
    # Test color mapping projection
    projector = ColorProjector()
    tex_mesh = projector.project_texture(mesh, rgb, intrinsics)
    assert tex_mesh.vertex_colors.shape == (225, 3)
    
    # Test Painter's Algorithm render synthesis
    generator = PoseGenerator(target_size=(112, 112))
    rendered = generator.render_pose(tex_mesh, intrinsics, yaw_deg=30.0)
    assert rendered.shape == (112, 112, 3)
