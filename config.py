from pydantic import BaseModel

class RecognitionConfig(BaseModel):
    max_track_history: int = 10
    track_iou_threshold: float = 0.4
    track_max_age: float = 0.5  # seconds to keep a track active without new detections
