from typing import Optional
from sqlmodel import Field, SQLModel
from sred.models.base import TimestampMixin

class MemoryDoc(TimestampMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int # Scoped to a run
    path: str # e.g. "memory/summary.md" or "memory/decisions.md"
    content_md: str
    content_hash: str # sha256 of content_md
