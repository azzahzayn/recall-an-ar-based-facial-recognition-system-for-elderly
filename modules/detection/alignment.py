import cv2
import numpy as np
from common.schemas import Landmarks5pt

# Reference landmark positions in a 112x112 coordinate system (standard InsightFace layout)
REFERENCE_5POINTS = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.6963],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Mouth Left Corner
    [70.7299, 92.3655]   # Mouth Right Corner
], dtype=np.float32)

def align_face(rgb: np.ndarray, landmarks: Landmarks5pt, target_size=(112, 112)) -> np.ndarray:
    """
    Applies a similarity transformation (scale, rotate, translate) to crop and align 
    a face from an RGB image using 5-point landmarks.
    """
    src = landmarks.points.astype(np.float32)
    
    # Estimate the optimal 2D similarity transform matrix M (2x3)
    M, _ = cv2.estimateAffinePartial2D(src, REFERENCE_5POINTS)
    
    if M is None:
        # Fallback: simple center-crop if affine estimation fails
        h, w = rgb.shape[:2]
        # Calculate bounding box of landmarks to crop
        min_xy = np.min(src, axis=0)
        max_xy = np.max(src, axis=0)
        center = ((min_xy + max_xy) / 2).astype(int)
        radius = int(np.max(max_xy - min_xy) * 0.7)
        
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(w, center[0] + radius)
        y2 = min(h, center[1] + radius)
        
        crop = rgb[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, target_size)
        else:
            return cv2.resize(rgb, target_size)
            
    # Warp image to normalized coordinate system
    warped = cv2.warpAffine(rgb, M, target_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped
