import os
import pytest
import numpy as np
from common.schemas import Embedding
from modules.database.config import DatabaseConfig
from modules.database.repository import IdentityRepository

def test_identity_repository(tmp_path):
    db_file = os.path.join(tmp_path, "test_identities.db")
    config = DatabaseConfig(db_path=db_file, similarity_threshold=0.5, match_strategy="min_distance")
    repo = IdentityRepository(config)
    
    # Create two orthogonal vector embeddings
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    emb1 = Embedding(vector=v1, model_version="test")
    
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    emb2 = Embedding(vector=v2, model_version="test")
    
    # Enroll identities
    repo.enroll_identity("id_alice", "Alice", [emb1], ["frontal"])
    repo.enroll_identity("id_bob", "Bob", [emb2], ["frontal"])
    
    # Query with a vector similar to Alice's embedding
    q1 = np.zeros(512, dtype=np.float32)
    q1[0] = 0.98
    q1[1] = 0.1
    q1 = q1 / np.linalg.norm(q1)
    q_emb1 = Embedding(vector=q1, model_version="test")
    
    name, score = repo.search_nearest(q_emb1)
    assert name == "Alice"
    assert score > 0.95
    
    # Query with an unknown vector (orthogonal to both)
    q_unk = np.zeros(512, dtype=np.float32)
    q_unk[10] = 1.0
    q_emb_unk = Embedding(vector=q_unk, model_version="test")
    
    name_unk, score_unk = repo.search_nearest(q_emb_unk)
    assert name_unk is None
    assert score_unk == 0.0
    
    # Test deleting an identity
    repo.delete_identity("id_alice")
    name_del, score_del = repo.search_nearest(q_emb1)
    assert name_del is None
    
    repo.close()

def test_identity_repository_model_version_mismatch(tmp_path):
    db_file = os.path.join(tmp_path, "test_identities_mismatch.db")
    config = DatabaseConfig(db_path=db_file, similarity_threshold=0.5)
    repo = IdentityRepository(config)
    
    # Alice enrolled with model 'model_A'
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    emb_alice = Embedding(vector=v1, model_version="model_A")
    repo.enroll_identity("id_alice", "Alice", [emb_alice], ["frontal"])
    
    # Query with a vector identical to Alice's embedding, but with model 'model_B'
    q1 = np.zeros(512, dtype=np.float32)
    q1[0] = 1.0
    q_emb_mismatch = Embedding(vector=q1, model_version="model_B")
    
    name, score = repo.search_nearest(q_emb_mismatch)
    # Since model versions mismatch, Alice should not be recognized
    assert name is None
    assert score == 0.0
    
    repo.close()

