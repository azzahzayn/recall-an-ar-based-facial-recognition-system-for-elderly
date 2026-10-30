import os
import cv2
import numpy as np
import logging
from typing import Optional
from common.schemas import Embedding
from modules.embedding.config import EmbedderConfig

logger = logging.getLogger("face_ar_system.embedding")

class FaceEmbedder:
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()
        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_mock = self.config.use_mock

        if not self.is_mock:
            try:
                import onnxruntime as ort
                self.ort = ort
                self._load_model()
            except Exception as e:
                logger.warning(f"Could not load ONNX model: {e}. Falling back to deterministic mock embedder.")
                self.is_mock = True

    def _load_model(self):
        model_path = self.config.model_path
        if not model_path:
            # Check or create default cache path
            os.makedirs(self.config.cache_dir, exist_ok=True)
            model_path = os.path.join(self.config.cache_dir, "arcface.onnx")
            
            if not os.path.exists(model_path):
                # Try downloading from model_url
                logger.info(f"Downloading ArcFace model from {self.config.model_url} to {model_path}...")
                import urllib.request
                try:
                    urllib.request.urlretrieve(self.config.model_url, model_path)
                    logger.info("Download completed successfully.")
                except Exception as ex:
                    logger.error(f"Failed to download model: {ex}.")
                    raise ex

        # Load session using CPUExecutionProvider for standard platform compatibility
        self.session = self.ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"ArcFace ONNX model loaded successfully from: {model_path}")

    def embed(self, face_crop: np.ndarray) -> Embedding:
        """
        Extracts a 512-D L2-normalized embedding from a 112x112 aligned face crop.
        """
        # Ensure correct crop dimensions
        if face_crop.shape[:2] != (112, 112):
            face_crop = cv2.resize(face_crop, (112, 112))

        if self.is_mock:
            vector = self._mock_embed(face_crop)
            version = "mock_v1"
        else:
            try:
                # Preprocess: ArcFace expects float32 image transposed to NCHW channel layout
                blob = face_crop.astype(np.float32)
                # Standard models trained with BGR input, transpose HWC -> CHW
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0) # Shape: (1, 3, 112, 112)
                
                # Execute ONNX session
                outputs = self.session.run([self.output_name], {self.input_name: blob})
                vector = outputs[0][0]
                
                # Normalize vector to unit sphere
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                version = "arcface_v10"
            except Exception as e:
                logger.error(f"ONNX inference failed: {e}. Using deterministic mock vector.")
                vector = self._mock_embed(face_crop)
                version = "mock_fallback"

        return Embedding(vector=vector, model_version=version)

    def _mock_embed(self, face_crop: np.ndarray) -> np.ndarray:
        """
        A deterministic mock projection mapping similar face pixels to similar 512-D vectors
        without loading heavy models.
        """
        resized = cv2.resize(face_crop, (8, 8))
        flat = resized.flatten().astype(np.float32)
        # Use fixed seed to guarantee deterministic results for testing
        state = np.random.RandomState(42)
        proj_matrix = state.randn(192, 512).astype(np.float32)
        embedding = np.dot(flat, proj_matrix)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding
