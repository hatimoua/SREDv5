from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


class TimestampMixin(SQLModel):
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
        nullable=False,
    )


class ProvenanceMixin(SQLModel):
    source_file_id: Optional[int] = Field(default=None, foreign_key="file.id")
    page_number: Optional[int] = Field(default=None)
    row_number: Optional[int] = Field(default=None)
