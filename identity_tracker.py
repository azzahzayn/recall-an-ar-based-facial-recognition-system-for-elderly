import time
from collections import Counter
from typing import List, Dict, Optional, Tuple
from common.schemas import BBox, RecognitionResult
from modules.recognition.config import RecognitionConfig

def compute_iou(box1: BBox, box2: BBox) -> float:
    """Computes Intersection-over-Union of two bounding boxes."""
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    
    union_area = float(box1_area + box2_area - inter_area)
    if union_area == 0:
        return 0.0
    return inter_area / union_area

class FaceTrack:
    def __init__(self, track_id: int, bbox: BBox, identity: Optional[str], confidence: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen = timestamp
        self.history: List[Optional[str]] = [identity]
        self.confidences: List[float] = [confidence]

    def update(self, bbox: BBox, identity: Optional[str], confidence: float, timestamp: float, max_history: int):
        self.bbox = bbox
        self.last_seen = timestamp
        self.history.append(identity)
        self.confidences.append(confidence)
        
        if len(self.history) > max_history:
            self.history.pop(0)
            self.confidences.pop(0)

    @property
    def smoothed_identity(self) -> Optional[str]:
        """Performs a majority vote on identity history to smooth output."""
        valid_votes = [name for name in self.history if name is not None]
        if not valid_votes:
            return None
        
        # If the majority of recent frames detected it as None/Unknown, mark as None
        none_count = self.history.count(None)
        if none_count > len(self.history) / 2:
            return None
            
        counter = Counter(valid_votes)
        return counter.most_common(1)[0][0]

    @property
    def average_confidence(self) -> float:
        return float(sum(self.confidences) / len(self.confidences))

class IdentityTracker:
    def __init__(self, config: Optional[RecognitionConfig] = None):
        self.config = config or RecognitionConfig()
        self.tracks: List[FaceTrack] = []
        self._next_track_id = 0

    def update(self, bboxes: List[BBox], identities: List[Optional[str]], confidences: List[float], timestamp: float) -> List[RecognitionResult]:
        """
        Associates frame detections with existing tracks and returns smoothed RecognitionResults.
        """
        # 1. Age out old tracks
        self.tracks = [t for t in self.tracks if (timestamp - t.last_seen) <= self.config.track_max_age]

        # 2. Greedy association
        matched_detections = set()
        matched_tracks = set()

        # Try to match incoming detections with tracks based on IoU
        associations: List[Tuple[float, int, int]] = [] # (IoU, track_index, detection_index)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, bbox in enumerate(bboxes):
                iou = compute_iou(track.bbox, bbox)
                if iou >= self.config.track_iou_threshold:
                    associations.append((iou, t_idx, d_idx))

        # Sort associations by IoU descending
        associations.sort(key=lambda x: x[0], reverse=True)

        for iou, t_idx, d_idx in associations:
            if t_idx not in matched_tracks and d_idx not in matched_detections:
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)
                # Update existing track
                self.tracks[t_idx].update(
                    bbox=bboxes[d_idx],
                    identity=identities[d_idx],
                    confidence=confidences[d_idx],
                    timestamp=timestamp,
                    max_history=self.config.max_track_history
                )

        # 3. Create new tracks for unmatched detections
        for d_idx, bbox in enumerate(bboxes):
            if d_idx not in matched_detections:
                new_track = FaceTrack(
                    track_id=self._next_track_id,
                    bbox=bbox,
                    identity=identities[d_idx],
                    confidence=confidences[d_idx],
                    timestamp=timestamp
                )
                self._next_track_id += 1
                self.tracks.append(new_track)

        # 4. Generate results for this frame (only for tracks updated in the current frame)
        results = []
        for track in self.tracks:
            if track.last_seen == timestamp:
                results.append(RecognitionResult(
                    bbox=track.bbox,
                    identity=track.smoothed_identity,
                    confidence=track.average_confidence,
                    timestamp=timestamp
                ))
        return results
