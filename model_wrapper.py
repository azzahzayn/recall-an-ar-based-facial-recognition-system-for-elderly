import cv2
import numpy as np
from typing import Tuple, Optional
from common.schemas import FaceMesh, Landmarks5pt, BBox

class ParametricFaceModel:
    def __init__(self, grid_res: int = 15):
        self.grid_res = grid_res
        self.vertices, self.faces = self._generate_template()
        
        # Define indices of the 5 key landmarks on our grid
        # (Assuming rows represent y, cols represent x in the grid)
        # Left Eye: Row 10, Col 4
        # Right Eye: Row 10, Col 10
        # Nose Tip: Row 7, Col 7 (Center)
        # Mouth Left: Row 4, Col 5
        # Mouth Right: Row 4, Col 9
        self.landmark_indices = [
            10 * grid_res + 4,   # Left Eye
            10 * grid_res + 10,  # Right Eye
            7 * grid_res + 7,    # Nose Tip
            4 * grid_res + 5,    # Mouth Left
            4 * grid_res + 9     # Mouth Right
        ]

    def _generate_template(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates an ellipsoid-based face mesh grid template."""
        res = self.grid_res
        u = np.linspace(-1.0, 1.0, res)
        v = np.linspace(-1.0, 1.0, res)
        uu, vv = np.meshgrid(u, v)
        
        # Flatten for easy indexing
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()
        
        # Create coordinates on an ellipsoid surface
        x = uu_flat * 0.12 # Standard face semi-width (~12cm)
        y = vv_flat * 0.16 # Standard face semi-height (~16cm)
        
        # Ellipsoid depth computation
        z_sq = 1.0 - (uu_flat)**2 - (vv_flat)**2
        z = np.zeros_like(z_sq)
        valid = z_sq > 0
        z[valid] = np.sqrt(z_sq[valid]) * 0.10 # Standard face depth projection (~10cm)
        
        # Align nose tip (center) towards positive Z
        vertices = np.stack([x, y, z], axis=-1)
        
        # Generate faces (triangulation of grid)
        faces = []
        for r in range(res - 1):
            for c in range(res - 1):
                # Node indices
                i1 = r * res + c
                i2 = r * res + (c + 1)
                i3 = (r + 1) * res + c
                i4 = (r + 1) * res + (c + 1)
                
                # Triangulate quad
                faces.append([i1, i2, i3])
                faces.append([i2, i4, i3])
                
        return vertices.astype(np.float32), np.array(faces, dtype=np.int32)

    def get_mesh(self, shape_params: np.ndarray, expression_params: np.ndarray) -> FaceMesh:
        """
        Morphs the template mesh based on shape and expression parameters.
        """
        morphed_vertices = self.vertices.copy()
        
        # Shape param 0: Width scale
        morphed_vertices[:, 0] *= (1.0 + shape_params[0] * 0.2)
        # Shape param 1: Height scale
        morphed_vertices[:, 1] *= (1.0 + shape_params[1] * 0.2)
        # Shape param 2: Depth scale
        morphed_vertices[:, 2] *= (1.0 + shape_params[2] * 0.2)
        
        # Expression param 0: Mouth Open
        # Apply vertical shift to mouth region (rows 3 to 5)
        res = self.grid_res
        for r in range(3, 6):
            for c in range(4, 11):
                idx = r * res + c
                # Translate downward
                morphed_vertices[idx, 1] -= expression_params[0] * 0.02
                
        # Expression param 1: Smile
        # Pull mouth corners (indices 4*res+5 and 4*res+9) outward and upward
        m_left_idx = 4 * res + 5
        m_right_idx = 4 * res + 9
        morphed_vertices[m_left_idx, 0] -= expression_params[1] * 0.015
        morphed_vertices[m_left_idx, 1] += expression_params[1] * 0.01
        morphed_vertices[m_right_idx, 0] += expression_params[1] * 0.015
        morphed_vertices[m_right_idx, 1] += expression_params[1] * 0.01

        return FaceMesh(
            vertices=morphed_vertices,
            faces=self.faces,
            shape_params=shape_params,
            expression_params=expression_params
        )

class FaceReconstructor:
    def __init__(self):
        self.model = ParametricFaceModel()

    def reconstruct(self, landmarks: Landmarks5pt, intrinsics: np.ndarray) -> Tuple[FaceMesh, np.ndarray, np.ndarray]:
        """
        Fits the 3D model to 2D landmarks and returns the reconstructed mesh in camera space,
        along with rotation vector rvec and translation vector tvec.
        """
        # 1. Retrieve 3D coordinates of key landmarks on template
        # Default shape & expression params = 0.0
        shape_params = np.zeros(3, dtype=np.float32)
        expr_params = np.zeros(2, dtype=np.float32)
        template_mesh = self.model.get_mesh(shape_params, expr_params)
        
        model_pts = template_mesh.vertices[self.model.landmark_indices]
        image_pts = landmarks.points.astype(np.float32)

        # 2. Solve Perspective-n-Point to find 3D pose (rotation and translation)
        dist_coeffs = np.zeros(4, dtype=np.float32) # Assume zero lens distortion for alignment
        
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=model_pts,
            imagePoints=image_pts,
            cameraMatrix=intrinsics,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        
        if not success:
            # Fallback values if solvePnP fails
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float32) # 1 meter away

        # 3. Transform all mesh vertices from object/template space into camera coordinate frame
        R, _ = cv2.Rodrigues(rvec)
        # Vertices in camera space: v_c = R * v_o + t
        cam_vertices = (template_mesh.vertices @ R.T) + tvec.T
        
        reconstructed_mesh = FaceMesh(
            vertices=cam_vertices.astype(np.float32),
            faces=template_mesh.faces,
            shape_params=shape_params,
            expression_params=expr_params,
            pose_params=np.concatenate([rvec.flatten(), tvec.flatten()])
        )
        
        return reconstructed_mesh, rvec, tvec
