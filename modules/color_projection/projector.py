import numpy as np
from common.schemas import FaceMesh, TexturedFaceMesh, CameraIntrinsics

class ColorProjector:
    def __init__(self):
        pass

    def project_texture(self, mesh: FaceMesh, rgb_image: np.ndarray, intrinsics: CameraIntrinsics) -> TexturedFaceMesh:
        """
        Projects RGB pixels from the 2D image onto the 3D face mesh vertices.
        Uses normal-based back-face culling to handle visibility.
        """
        h, w = rgb_image.shape[:2]
        vertices = mesh.vertices  # (V, 3) in camera space
        faces = mesh.faces        # (F, 3)
        
        # 1. Project 3D vertices to 2D image plane (pinhole camera projection)
        X = vertices[:, 0]
        Y = vertices[:, 1]
        Z = vertices[:, 2]
        
        # Safeguard division by zero
        Z_safe = np.where(Z == 0.0, 1e-5, Z)
        
        u = (X * intrinsics.fx / Z_safe) + intrinsics.cx
        v = (Y * intrinsics.fy / Z_safe) + intrinsics.cy
        
        # 2. Compute vertex normals for back-face visibility check
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Compute face normal vectors via cross product
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = np.where(norms > 0, face_normals / norms, 0.0)
        
        # Accumulate face normals to compute vertex normals (averaging neighboring faces)
        v_normals = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(v_normals, faces[:, i], face_normals)
            
        v_norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
        v_normals = np.where(v_norms > 0, v_normals / v_norms, 0.0)
        
        # Compute look/view vectors pointing from camera (0, 0, 0) to each vertex
        v_view = vertices.copy()
        v_view_norms = np.linalg.norm(v_view, axis=1, keepdims=True)
        v_view = np.where(v_view_norms > 0, v_view / v_view_norms, 0.0)
        
        # Dot product of normal vector and viewing vector
        # A dot product > 0 means the normal points away from the camera (facing away/occluded)
        dot_products = np.sum(v_normals * v_view, axis=1)
        is_visible = dot_products < 0.1  # Threshold slightly offset to preserve border edges
        
        # 3. Sample colors
        vertex_colors = np.zeros((len(vertices), 3), dtype=np.uint8)
        
        # Default skin color fallback for back-facing/out-of-bounds vertices
        default_skin = np.array([215, 175, 135], dtype=np.uint8)
        
        for i in range(len(vertices)):
            if not is_visible[i]:
                vertex_colors[i] = default_skin
                continue
                
            ui, vi = int(round(u[i])), int(round(v[i]))
            
            # Verify coordinates fall inside RGB boundaries
            if 0 <= ui < w and 0 <= vi < h:
                vertex_colors[i] = rgb_image[vi, ui]
            else:
                vertex_colors[i] = default_skin
                
        return TexturedFaceMesh(mesh=mesh, vertex_colors=vertex_colors)
