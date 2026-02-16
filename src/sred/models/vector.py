from enum import Enum
from typing import Optional, Any
from sqlmodel import Field, SQLModel, UniqueConstraint
from sred.models.base import TimestampMixin
import numpy as np

class EntityType(str, Enum):
    SEGMENT = "SEGMENT"
    STAGING_ROW = "STAGING_ROW"
    MEMORY_MD = "MEMORY_MD"

class VectorEmbedding(TimestampMixin, table=True):
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", "model", name="unique_embedding_per_model"),
    )
    
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True) # Indexed for scoped search
    
    entity_type: EntityType
    entity_id: int
    
    model: str
    dims: int
    vector: bytes # Store as BLOB (numpy tobytes)
    text_hash: str # For caching
    
    def set_vector(self, embedding: list[float]):
        """Convert list of floats to bytes for storage."""
        arr = np.array(embedding, dtype=np.float32)
        self.vector = arr.tobytes()
        self.dims = len(embedding)

    def get_vector(self) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return np.frombuffer(self.vector, dtype=np.float32)
