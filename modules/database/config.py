from pydantic import BaseModel
from typing import Literal

class DatabaseConfig(BaseModel):
    db_path: str = "face_identities.db"
    # Cosine similarity threshold. Since vectors are L2-normalized, 
    # dot product ranges from -1.0 to 1.0. A score of 0.6 is a standard threshold.
    similarity_threshold: float = 0.6
    match_strategy: Literal["min_distance", "centroid"] = "min_distance"
