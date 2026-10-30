import numpy as np
import pytest
from modules.embedding.config import EmbedderConfig
from modules.embedding.embedder import FaceEmbedder

def test_mock_embedder():
    config = EmbedderConfig(use_mock=True)
    embedder = FaceEmbedder(config)
    
    # Create two distinct face crops
    img1 = np.ones((112, 112, 3), dtype=np.uint8) * 80
    img2 = np.ones((112, 112, 3), dtype=np.uint8) * 180
    img1[10:30, 10:30] = 0
    img2[40:60, 40:60] = 0

    
    emb1 = embedder.embed(img1)
    emb2 = embedder.embed(img2)
    
    # Ensure vector dimensions and normalized output lengths are accurate
    assert emb1.vector.shape == (512,)
    assert emb2.vector.shape == (512,)
    assert np.allclose(np.linalg.norm(emb1.vector), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(emb2.vector), 1.0, atol=1e-5)
    
    # Assert determinism (identical input must yield identical output)
    emb1_retry = embedder.embed(img1)
    assert np.allclose(emb1.vector, emb1_retry.vector)
    
    # Assert distinct inputs yield distinct outputs
    similarity = np.dot(emb1.vector, emb2.vector)
    assert similarity < 0.9999
