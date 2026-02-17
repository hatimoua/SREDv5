"""
World-model entities for Increment 8:
- Contradiction  (severity-tagged data conflicts)
- ReviewTask     (actionable items for human review, deduped by issue_key)
- ReviewDecision (human resolution of a ReviewTask)
- DecisionLock   (permanent record preventing re-opening of resolved issues)
"""
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel, UniqueConstraint
from sred.models.base import TimestampMixin


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------
class ContradictionSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKING = "BLOCKING"


class ContradictionType(str, Enum):
    MISSING_RATE = "MISSING_RATE"
    PAYROLL_MISMATCH = "PAYROLL_MISMATCH"
    UNKNOWN_BASIS = "UNKNOWN_BASIS"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    OTHER = "OTHER"


class ContradictionStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class Contradiction(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)

    issue_key: str = Field(index=True, description="Unique key for deduplication within a run, e.g. 'MISSING_RATE:person:3'")
    contradiction_type: ContradictionType
    severity: ContradictionSeverity
    description: str

    status: ContradictionStatus = Field(default=ContradictionStatus.OPEN)

    # Optional FK pointers for provenance
    related_entity_type: Optional[str] = None  # e.g. "Person", "StagingRow"
    related_entity_id: Optional[int] = None


# ---------------------------------------------------------------------------
# ReviewTask
# ---------------------------------------------------------------------------
class ReviewTaskStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    SUPERSEDED = "SUPERSEDED"


class ReviewTask(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)

    issue_key: str = Field(index=True, description="Dedup key, e.g. 'MISSING_RATE:person:3'")
    title: str
    description: str
    severity: ContradictionSeverity = Field(default=ContradictionSeverity.MEDIUM)

    status: ReviewTaskStatus = Field(default=ReviewTaskStatus.OPEN)

    # Link to the contradiction that spawned this task (optional)
    contradiction_id: Optional[int] = Field(default=None, foreign_key="contradiction.id")


# ---------------------------------------------------------------------------
# ReviewDecision
# ---------------------------------------------------------------------------
class ReviewDecision(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)

    task_id: int = Field(foreign_key="reviewtask.id")
    decision: str  # Free-text human decision
    decided_by: str = Field(default="HUMAN")  # HUMAN or SYSTEM


# ---------------------------------------------------------------------------
# DecisionLock
# ---------------------------------------------------------------------------
class DecisionLock(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)

    issue_key: str = Field(index=True, description="Same key space as ReviewTask.issue_key")
    decision_id: int = Field(foreign_key="reviewdecision.id")
    reason: str

    active: bool = Field(default=True)
