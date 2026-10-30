import cv2
import numpy as np
from typing import List, Tuple
from common.schemas import TexturedFaceMesh, CameraIntrinsics, Landmarks5pt

class PoseGenerator:
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size

    def render_pose(
        self,
        textured_mesh: TexturedFaceMesh,
        intrinsics: CameraIntrinsics,
        yaw_deg: float,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0
    ) -> np.ndarray:
        """
        Renders the textured 3D face mesh from a custom virtual camera angle.
        Uses the Painter's Algorithm for occlusion-correct rendering.
        """
        mesh = textured_mesh.mesh
        vertices = mesh.vertices.copy()  # (V, 3) in camera space
        colors = textured_mesh.vertex_colors  # (V, 3)
        faces = mesh.faces  # (F, 3)

        # 1. Convert Euler angles to radians
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        
        # Y-axis rotation (Yaw)
        R_yaw = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ], dtype=np.float32)
        
        # X-axis rotation (Pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ], dtype=np.float32)
        
        # Z-axis rotation (Roll)
        R_roll = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combined rotation matrix: R = Rz * Rx * Ry
        R = R_roll @ R_pitch @ R_yaw

        # 2. Rotate vertices around the nose tip (index 112 in 15x15 model)
        nose_idx = 112
        if len(vertices) > nose_idx:
            center = vertices[nose_idx].copy()
        else:
            center = np.mean(vertices, axis=0)
            
        # Shift center to origin, rotate, then shift back
        vertices_rotated = (vertices - center) @ R.T + center

        # 3. Project 3D rotated vertices back to 2D screen coordinates
        X = vertices_rotated[:, 0]
        Y = vertices_rotated[:, 1]
        Z = vertices_rotated[:, 2]
        
        # Safe divide by depth
        Z_safe = np.where(Z == 0.0, 1e-5, Z)
        
        u = (X * intrinsics.fx / Z_safe) + intrinsics.cx
        v = (Y * intrinsics.fy / Z_safe) + intrinsics.cy
        
        # 4. Painter's Algorithm: sort faces by average depth in descending order (back-to-front)
        face_depths = np.mean(Z[faces], axis=1)
        sorted_face_indices = np.argsort(face_depths)[::-1]
        
        # 5. Draw faces onto a black canvas matching camera intrinsics
        canvas = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
        
        for f_idx in sorted_face_indices:
            face = faces[f_idx]
            
            # Extract 2D points of the triangle vertices
            pts = np.stack([u[face], v[face]], axis=1).astype(np.int32)
            
            # Prune out-of-bounds triangles that cause rendering artifacts
            if (np.any(pts < -200) or 
                np.any(pts[:, 0] > intrinsics.width + 200) or 
                np.any(pts[:, 1] > intrinsics.height + 200)):
                continue
                
            # Compute average color of face vertices
            color = np.mean(colors[face], axis=0).astype(int)
            
            # Fill convex polygon/triangle
            cv2.fillConvexPoly(canvas, pts, (int(color[0]), int(color[1]), int(color[2])), cv2.LINE_AA)
            
        # 6. Align and crop the face region using 5-point landmark alignment
        if len(vertices) == 225:
            # Landmark indices corresponding to Left Eye, Right Eye, Nose Tip, Mouth Left, Mouth Right in 15x15 grid
            landmark_indices = [154, 160, 112, 65, 69]
            pts_2d = np.stack([u[landmark_indices], v[landmark_indices]], axis=1)
            landmarks_5pt = Landmarks5pt(points=pts_2d)
            
            from modules.detection.alignment import align_face
            aligned_crop = align_face(canvas, landmarks_5pt, target_size=self.target_size)
            return aligned_crop
            
        # Fallback to simple bounding box center crop if vertex count is different
        u_face = u[mesh.vertices[:, 2] > 0.0]
        v_face = v[mesh.vertices[:, 2] > 0.0]
        
        if len(u_face) > 0:
            umin, umax = int(np.min(u_face)), int(np.max(u_face))
            vmin, vmax = int(np.min(v_face)), int(np.max(v_face))
            
            # Apply padding margin around face crop bounds
            pad_w = int((umax - umin) * 0.15)
            pad_h = int((vmax - vmin) * 0.15)
            
            x1 = max(0, umin - pad_w)
            y1 = max(0, vmin - pad_h)
            x2 = min(intrinsics.width, umax + pad_w)
            y2 = min(intrinsics.height, vmax + pad_h)
            
            face_crop = canvas[y1:y2, x1:x2]
            if face_crop.size > 0:
                return cv2.resize(face_crop, self.target_size)
                
        return cv2.resize(canvas, self.target_size)

    def generate_pose_gallery(self, textured_mesh: TexturedFaceMesh, intrinsics: CameraIntrinsics) -> List[Tuple[np.ndarray, float]]:
        """
        Generates standard multi-angle pose gallery: yaws [-45, -30, -15, 0, 15, 30, 45] degrees.
        Returns a list of tuples (rendered_crop, yaw_deg).
        """
        yaws = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
        gallery = []
        for yaw in yaws:
            rendered = self.render_pose(textured_mesh, intrinsics, yaw_deg=yaw)
            gallery.append((rendered, yaw))
        return gallery
