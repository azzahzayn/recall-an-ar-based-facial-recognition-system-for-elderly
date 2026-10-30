import numpy as np
from typing import Optional, Tuple
from scipy.spatial import KDTree
from common.schemas import FramePacket, PointCloud, BBox

class PointCloudProcessor:
    def __init__(self, max_depth: float = 2.0, min_depth: float = 0.2):
        self.max_depth = max_depth
        self.min_depth = min_depth

    def back_project(self, packet: FramePacket, bbox: Optional[BBox] = None) -> PointCloud:
        """
        Converts the depth frame (optionally restricted to a face bounding box) 
        into a 3D point cloud using the camera intrinsics.
        """
        depth = packet.depth
        rgb = packet.rgb
        intr = packet.intrinsics

        h, w = depth.shape
        v_indices, u_indices = np.indices((h, w))

        # Create validity mask based on depth thresholds
        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)

        # Apply face bounding box crop if provided
        if bbox:
            ymin = max(0, int(bbox.ymin))
            ymax = min(h, int(bbox.ymax))
            xmin = max(0, int(bbox.xmin))
            xmax = min(w, int(bbox.xmax))
            
            bbox_mask = np.zeros_like(valid_mask)
            bbox_mask[ymin:ymax, xmin:xmax] = True
            valid_mask = valid_mask & bbox_mask

        # Extract active pixels
        z = depth[valid_mask]
        u = u_indices[valid_mask]
        v = v_indices[valid_mask]

        if len(z) == 0:
            return PointCloud(points=np.empty((0, 3), dtype=np.float32), colors=np.empty((0, 3), dtype=np.uint8))

        # Pinhole projection equations
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        
        points = np.stack([x, y, z], axis=-1).astype(np.float32)
        colors = rgb[valid_mask].astype(np.uint8)

        return PointCloud(points=points, colors=colors)

    def remove_statistical_outliers(self, pc: PointCloud, k: int = 15, std_ratio: float = 2.0) -> PointCloud:
        """
        Filters noise from the point cloud by calculating neighbor distance statistics via SciPy's KDTree.
        Identical function to Open3D's StatisticalOutlierRemoval.
        """
        points = pc.points
        colors = pc.colors

        if len(points) <= k:
            return pc

        # Construct space-partitioning tree
        tree = KDTree(points)
        
        # Query distances of k+1 nearest neighbors (index 0 is the point itself)
        distances, _ = tree.query(points, k=k + 1, workers=-1)
        
        # Compute mean neighbor distance for each point (excluding distance to self)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        cutoff = global_mean + std_ratio * global_std
        inlier_mask = mean_distances < cutoff

        return PointCloud(
            points=points[inlier_mask],
            colors=colors[inlier_mask]
        )
