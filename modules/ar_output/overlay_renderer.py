import cv2
import numpy as np
from typing import List
from common.schemas import RecognitionResult

class OverlayRenderer:
    def __init__(self, target_width: int = 640, target_height: int = 480):
        self.target_width = target_width
        self.target_height = target_height

    def render(self, rgb_image: np.ndarray, results: List[RecognitionResult]) -> np.ndarray:
        """
        Draws premium AR-HUD bounding corner brackets and identity tags over an RGB frame.
        """
        # Ensure working on a mutable copy
        img = rgb_image.copy()
        
        for res in results:
            box = res.bbox
            name = res.identity
            confidence = res.confidence
            
            # Curate color palettes (RGB format)
            if name:
                # Neon cyan/teal for recognized identities
                color_rgb = (0, 240, 200)
                label = f"{name} ({confidence*100:.0f}%)"
            else:
                # Coral/Orange-yellow for unknown subjects
                color_rgb = (255, 140, 0)
                label = "Unknown"
                
            x1, y1, x2, y2 = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
            
            # Draw subtle thin tracking box
            cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 1, cv2.LINE_AA)
            
            # Draw premium thick corner brackets (HUD look)
            length = min(18, int((x2 - x1) * 0.15))
            thickness = 2
            
            # Top-left corner
            cv2.line(img, (x1, y1), (x1 + length, y1), color_rgb, thickness)
            cv2.line(img, (x1, y1), (x1, y1 + length), color_rgb, thickness)
            # Top-right corner
            cv2.line(img, (x2, y1), (x2 - length, y1), color_rgb, thickness)
            cv2.line(img, (x2, y1), (x2, y1 + length), color_rgb, thickness)
            # Bottom-left corner
            cv2.line(img, (x1, y2), (x1 + length, y2), color_rgb, thickness)
            cv2.line(img, (x1, y2), (x1, y2 - length), color_rgb, thickness)
            # Bottom-right corner
            cv2.line(img, (x2, y2), (x2 - length, y2), color_rgb, thickness)
            cv2.line(img, (x2, y2), (x2, y2 - length), color_rgb, thickness)
            
            # Render label above face box
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.45
            font_thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Bounds for the background panel
            bg_x1 = x1
            bg_y1 = max(0, y1 - text_h - 10)
            bg_x2 = min(img.shape[1], x1 + text_w + 10)
            bg_y2 = y1
            
            # Semi-transparent backdrop block
            overlay = img.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            alpha = 0.55
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
            
            # Overlay labeled name text
            cv2.putText(img, label, (x1 + 5, y1 - 6), font, font_scale, color_rgb, font_thickness, cv2.LINE_AA)
            
        return img
