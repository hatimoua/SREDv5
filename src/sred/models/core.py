from enum import Enum, auto
from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel
from datetime import datetime
from sred.models.base import TimestampMixin, ProvenanceMixin

class RunStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    PROCESSING = "PROCESSING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class SegmentStatus(str, Enum):
    PENDING = "PENDING"
    DONE = "DONE"
    ERROR = "ERROR"

class Run(TimestampMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    status: RunStatus = Field(default=RunStatus.INITIALIZING)
    
    files: List["File"] = Relationship(back_populates="run")

class RateStatus(str, Enum):
    PENDING = "PENDING"
    SET = "SET"

class Person(TimestampMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    
    name: str
    role: str = Field(description="Role is now required")
    email: Optional[str] = None
    
    hourly_rate: Optional[float] = Field(default=None)
    rate_status: RateStatus = Field(default=RateStatus.PENDING)

class FileStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSED = "PROCESSED"
    ERROR = "ERROR"

class File(TimestampMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    
    path: str = Field(description="Stored path relative to APP_DATA_DIR")
    original_filename: str
    file_type: str # This is mime_type now, keeping name or renaming? Let's clarify. 
    # User asked for mime_type. Let's add mime_type and keep file_type as legacy or alias? 
    # Actually, let's just use mime_type and map file_type to it if needed, or keep both.
    # The user instruction says: "File ... status, mime_type, size_bytes".
    mime_type: str
    size_bytes: int
    status: FileStatus = Field(default=FileStatus.UPLOADED)
    
    content_hash: str
    
    run: Optional[Run] = Relationship(back_populates="files")
    segments: List["Segment"] = Relationship(back_populates="file", sa_relationship_kwargs={"foreign_keys": "Segment.file_id"})

class Segment(TimestampMixin, ProvenanceMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="file.id")
    # Denormalized run_id for scope efficiency
    run_id: int = Field(foreign_key="run.id", index=True)
    
    content: str
    status: SegmentStatus = Field(default=SegmentStatus.PENDING)
    
    file: Optional[File] = Relationship(back_populates="segments", sa_relationship_kwargs={"foreign_keys": "Segment.file_id"})
    # Artifacts now link more loosely or via join table, but we'll remove the strict back_populates for now
    # or keep it if we update Artifact as well. 
    # For now, let's keep the relationship but we will update Artifact side next.
