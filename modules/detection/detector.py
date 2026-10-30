import cv2
import numpy as np
import logging
from typing import List, Optional
from common.schemas import BBox, Landmarks5pt, DetectionResult
from modules.detection.config import DetectorConfig

logger = logging.getLogger("face_ar_system.detection")

class FaceDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.use_mediapipe = False
        self.mp_face_mesh = None
        self.face_cascade = None

        # Try to initialize MediaPipe Face Mesh
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_detection_confidence
            )
            self.use_mediapipe = True
            logger.info("MediaPipe Face Mesh initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe Face Mesh: {e}. Falling back to Haar Cascades.")

        # Fallback setup: Load OpenCV Haar Cascade
        if not self.use_mediapipe or self.config.fallback_to_cascade:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                import os
                local_dir = os.path.dirname(os.path.abspath(__file__))
                local_path = os.path.join(local_dir, "haarcascade_frontalface_default.xml")
                if os.path.exists(local_path):
                    self.face_cascade = cv2.CascadeClassifier(local_path)
            if self.face_cascade.empty():
                logger.error("Failed to load OpenCV frontal face Haar cascade.")
            else:
                logger.info("OpenCV frontal face Haar cascade loaded successfully.")

    def detect(self, rgb: np.ndarray) -> DetectionResult:
        """
        Locates faces and returns bounding boxes with 5-point landmarks.
        """
        h, w = rgb.shape[:2]
        
        # 1. Primary path: MediaPipe Face Mesh
        if self.use_mediapipe and self.mp_face_mesh:
            try:
                results = self.mp_face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    boxes = []
                    landmarks_list = []
                    
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract landmark coordinates in pixels
                        pts = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                        
                        # Calculate bounding box
                        xmin, ymin = np.min(pts, axis=0)
                        xmax, ymax = np.max(pts, axis=0)
                        
                        # Apply slight margins
                        pad_w = (xmax - xmin) * 0.1
                        pad_h = (ymax - ymin) * 0.1
                        xmin_p = max(0, int(xmin - pad_w))
                        ymin_p = max(0, int(ymin - pad_h))
                        xmax_p = min(w, int(xmax + pad_w))
                        ymax_p = min(h, int(ymax + pad_h))
                        
                        # Extract 5 landmarks for alignment:
                        # Index 468: Left iris center (or index 33 left eye outer corner)
                        # Index 473: Right iris center (or index 263 right eye outer corner)
                        # Index 1: Nose tip
                        # Index 61: Mouth left corner
                        # Index 291: Mouth right corner
                        left_eye = pts[468] if len(pts) > 468 else pts[33]
                        right_eye = pts[473] if len(pts) > 473 else pts[263]
                        nose = pts[1]
                        mouth_left = pts[61]
                        mouth_right = pts[291]
                        
                        five_points = np.stack([left_eye, right_eye, nose, mouth_left, mouth_right])
                        
                        boxes.append(BBox(xmin=xmin_p, ymin=ymin_p, xmax=xmax_p, ymax=ymax_p, score=1.0))
                        landmarks_list.append(Landmarks5pt(points=five_points))
                        
                    return DetectionResult(boxes=boxes, landmarks=landmarks_list)
            except Exception as e:
                logger.error(f"Error during MediaPipe detection: {e}. Attempting cascade fallback.")

        # 2. Fallback path: OpenCV Haar Cascades
        if self.face_cascade and not self.face_cascade.empty():
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            boxes = []
            landmarks_list = []
            
            for (x, y, fw, fh) in faces:
                bbox = BBox(xmin=int(x), ymin=int(y), xmax=int(x+fw), ymax=int(y+fh), score=0.8)
                
                # Estimate standard face proportion landmarks
                # Left eye, Right eye, Nose tip, Mouth left, Mouth right
                left_eye = np.array([x + fw * 0.35, y + fh * 0.4], dtype=np.float32)
                right_eye = np.array([x + fw * 0.65, y + fh * 0.4], dtype=np.float32)
                nose = np.array([x + fw * 0.5, y + fh * 0.55], dtype=np.float32)
                mouth_left = np.array([x + fw * 0.35, y + fh * 0.75], dtype=np.float32)
                mouth_right = np.array([x + fw * 0.65, y + fh * 0.75], dtype=np.float32)
                
                five_points = np.stack([left_eye, right_eye, nose, mouth_left, mouth_right])
                
                boxes.append(bbox)
                landmarks_list.append(Landmarks5pt(points=five_points))
                
            return DetectionResult(boxes=boxes, landmarks=landmarks_list)
            
        return DetectionResult(boxes=[], landmarks=[])

    def close(self):
        if self.mp_face_mesh:
            self.mp_face_mesh.close()
