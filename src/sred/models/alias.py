from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel, UniqueConstraint
from sred.models.base import TimestampMixin


class AliasStatus(str, Enum):
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class PersonAlias(TimestampMixin, table=True):
    """Maps raw name variants (from CSV/timesheet) to canonical Person records.

    Unique constraint on (run_id, alias) prevents duplicate alias entries per run.
    """
    __table_args__ = (
        UniqueConstraint("run_id", "alias", name="uq_person_alias_run_alias"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    person_id: int = Field(foreign_key="person.id")

    alias: str = Field(description="Raw name variant found in source data, e.g. 'J. Doe'")
    source: Optional[str] = Field(default=None, description="Origin hint, e.g. 'timesheet.csv'")
    confidence: float = Field(default=0.0, description="Fuzzy match score 0-1")
    status: AliasStatus = Field(default=AliasStatus.PROPOSED)
