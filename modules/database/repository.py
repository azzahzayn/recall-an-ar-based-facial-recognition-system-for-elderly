import sqlite3
import numpy as np
import time
import logging
from typing import List, Optional, Tuple
from common.schemas import Embedding, Identity
from modules.database.config import DatabaseConfig

logger = logging.getLogger("face_ar_system.database")

class IdentityRepository:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        with self._conn:
            # Table to store main identity info
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    identity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            # Table to store individual embeddings (allowing multiple vectors per identity)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    model_version TEXT NOT NULL,
                    pose_label TEXT,
                    FOREIGN KEY(identity_id) REFERENCES identities(identity_id) ON DELETE CASCADE
                )
            """)
            logger.info(f"SQLite database initialized at: {self.config.db_path}")

    def enroll_identity(self, identity_id: str, name: str, embeddings: List[Embedding], pose_labels: Optional[List[str]] = None) -> None:
        """
        Enrolls a new identity with a gallery of one or more face embeddings.
        """
        with self._conn:
            # Delete if exists to support re-enrollment / update
            self._conn.execute("DELETE FROM identities WHERE identity_id = ?", (identity_id,))
            self._conn.execute("DELETE FROM embeddings WHERE identity_id = ?", (identity_id,))
            
            # Insert identity metadata
            self._conn.execute(
                "INSERT INTO identities (identity_id, name, created_at) VALUES (?, ?, ?)",
                (identity_id, name, time.time())
            )
            
            # Insert embeddings
            for idx, emb in enumerate(embeddings):
                pose = pose_labels[idx] if (pose_labels and idx < len(pose_labels)) else f"pose_{idx}"
                vector_bytes = emb.vector.astype(np.float32).tobytes()
                self._conn.execute(
                    "INSERT INTO embeddings (identity_id, vector, model_version, pose_label) VALUES (?, ?, ?, ?)",
                    (identity_id, vector_bytes, emb.model_version, pose)
                )
            logger.info(f"Enrolled identity: {name} ({identity_id}) with {len(embeddings)} embeddings.")

    def delete_identity(self, identity_id: str) -> None:
        """
        Deletes an identity and all corresponding embeddings.
        """
        with self._conn:
            self._conn.execute("DELETE FROM identities WHERE identity_id = ?", (identity_id,))
            self._conn.execute("DELETE FROM embeddings WHERE identity_id = ?", (identity_id,))
            logger.info(f"Deleted identity: {identity_id}")

    def search_nearest(self, query_emb: Embedding) -> Tuple[Optional[str], float]:
        """
        Queries the database for the closest identity using the similarity threshold.
        Returns (identity_name, similarity_score) if a match is found, otherwise (None, best_score).
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT e.identity_id, i.name, e.vector FROM embeddings e "
            "JOIN identities i ON e.identity_id = i.identity_id "
            "WHERE e.model_version = ?",
            (query_emb.model_version,)
        )
        rows = cursor.fetchall()
        
        if not rows:
            return None, 0.0

        # Extract vectors and labels
        identity_ids = []
        names = []
        vectors = []
        for identity_id, name, vector_bytes in rows:
            vec = np.frombuffer(vector_bytes, dtype=np.float32)
            if vec.shape[0] == 512:
                identity_ids.append(identity_id)
                names.append(name)
                vectors.append(vec)

        if not vectors:
            return None, 0.0

        # Stack into shape (NumEmbeddings, 512)
        db_matrix = np.stack(vectors)
        query_vec = query_emb.vector.astype(np.float32)
        
        # Calculate dot products (cosine similarity, since vectors are L2-normalized)
        similarities = np.dot(db_matrix, query_vec)

        best_identity = None
        best_score = 0.0

        if self.config.match_strategy == "min_distance":
            # Multi-vector strategy: match with maximum similarity to ANY enrolled pose
            unique_matches = {}
            for i in range(len(identity_ids)):
                id_val = identity_ids[i]
                name_val = names[i]
                score = similarities[i]
                if id_val not in unique_matches or score > unique_matches[id_val][1]:
                    unique_matches[id_val] = (name_val, score)

            # Find the globally highest similarity score
            logger.info("Database match search scores:")
            for id_val, (name_val, score) in unique_matches.items():
                logger.info(f"  - {name_val} ({id_val}): score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_identity = name_val

        elif self.config.match_strategy == "centroid":
            # Centroid strategy: compare query to mean vector of the subject's gallery
            unique_vectors = {}
            identity_names = {}
            for i in range(len(identity_ids)):
                id_val = identity_ids[i]
                identity_names[id_val] = names[i]
                if id_val not in unique_vectors:
                    unique_vectors[id_val] = []
                unique_vectors[id_val].append(vectors[i])

            for id_val, vecs in unique_vectors.items():
                centroid = np.mean(vecs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                score = np.dot(centroid, query_vec)
                if score > best_score:
                    best_score = score
                    best_identity = identity_names[id_val]

        if best_score >= self.config.similarity_threshold:
            return best_identity, float(best_score)
        else:
            return None, float(best_score)

    def get_all_identities(self) -> List[Identity]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT identity_id, name FROM identities")
        id_rows = cursor.fetchall()
        
        identities = []
        for identity_id, name in id_rows:
            cursor.execute("SELECT vector, model_version FROM embeddings WHERE identity_id = ?", (identity_id,))
            emb_rows = cursor.fetchall()
            embs = [
                Embedding(vector=np.frombuffer(vec_bytes, dtype=np.float32), model_version=mv)
                for vec_bytes, mv in emb_rows
            ]
            identities.append(Identity(identity_id=identity_id, name=name, embeddings=embs))
        return identities

    def close(self):
        if self._conn:
            self._conn.close()
