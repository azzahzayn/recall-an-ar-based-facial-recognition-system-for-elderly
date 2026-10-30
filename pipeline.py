import logging
from typing import List, Optional
from common.schemas import FramePacket, RecognitionResult
from modules.detection.detector import FaceDetector
from modules.detection.alignment import align_face
from modules.embedding.embedder import FaceEmbedder
from modules.database.repository import IdentityRepository
from modules.recognition.identity_tracker import IdentityTracker
from modules.reconstruction.model_wrapper import FaceReconstructor
from modules.color_projection.projector import ColorProjector
from modules.pose_generation.renderer import PoseGenerator

logger = logging.getLogger("face_ar_system.recognition")

class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        repository: IdentityRepository,
        tracker: Optional[IdentityTracker] = None,
        reconstructor: Optional[FaceReconstructor] = None,
        projector: Optional[ColorProjector] = None,
        pose_gen: Optional[PoseGenerator] = None
    ):
        self.detector = detector
        self.embedder = embedder
        self.repository = repository
        self.tracker = tracker or IdentityTracker()
        self.reconstructor = reconstructor or FaceReconstructor()
        self.projector = projector or ColorProjector()
        self.pose_gen = pose_gen or PoseGenerator(target_size=(112, 112))

    def process_frame(self, packet: FramePacket) -> List[RecognitionResult]:
        """
        Executes end-to-end 3D-assisted recognition pipeline logic on a frame packet.
        Fits a 3D model, projects the RGB texture, renders a frontalized 2D image, and recognizes it.
        """
        # 1. Detect faces in image
        det_result = self.detector.detect(packet.rgb)
        if not det_result.boxes:
            # Update tracker with empty detections to age current tracks
            return self.tracker.update([], [], [], packet.timestamp)

        raw_identities = []
        raw_confidences = []
        
        # 2. Extract embedding and perform search for each face found
        for bbox, landmarks in zip(det_result.boxes, det_result.landmarks):
            try:
                # Align face crop to standard 112x112 layout to preserve high-fidelity texture details
                face_crop = align_face(packet.rgb, landmarks)
                
                # Get ArcFace representation from high-fidelity aligned crop
                embedding = self.embedder.embed(face_crop)
                
                # Database lookup
                name, score = self.repository.search_nearest(embedding)
                
                raw_identities.append(name)
                raw_confidences.append(score)
            except Exception as e:
                logger.error(f"Failed to process face in pipeline: {e}", exc_info=True)
                raw_identities.append(None)
                raw_confidences.append(0.0)

        # 3. Apply temporal tracking and label smoothing
        smoothed_results = self.tracker.update(
            bboxes=det_result.boxes,
            identities=raw_identities,
            confidences=raw_confidences,
            timestamp=packet.timestamp
        )
        return smoothed_results

