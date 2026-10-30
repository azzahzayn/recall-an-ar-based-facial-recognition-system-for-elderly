from pydantic import BaseModel

class DetectorConfig(BaseModel):
    min_detection_confidence: float = 0.5
    model_selection: int = 0  # 0 for short-range, 1 for full-range (MediaPipe specific)
    fallback_to_cascade: bool = True
