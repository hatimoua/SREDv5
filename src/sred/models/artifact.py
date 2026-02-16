from enum import Enum
from typing import Optional, TYPE_CHECKING
from sqlmodel import Field, SQLModel, Relationship
from sred.models.base import TimestampMixin, ProvenanceMixin

if TYPE_CHECKING:
    from sred.models.core import Segment

class ArtifactKind(str, Enum):
    VISION_TEXT = "VISION_TEXT"
    VISION_TABLES_JSON = "VISION_TABLES_JSON"
    # Future kinds: ENTITY_EXTRACT, TECHNICAL_BLOCK

class ExtractionArtifact(TimestampMixin, ProvenanceMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="file.id")
    run_id: int = Field(foreign_key="run.id", index=True)
    
    kind: ArtifactKind
    data: str # JSON string or raw text depending on kind
    model: str # e.g. gpt-4o
    confidence: Optional[float] = None
    
    # Store list of segment IDs as JSON string e.g. "[1, 2, 3]"
    segment_ids_json: Optional[str] = None
    
    # Relationships
    file: Optional["File"] = Relationship(sa_relationship_kwargs={"foreign_keys": "ExtractionArtifact.file_id"})
