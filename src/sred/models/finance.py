from datetime import date
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel, UniqueConstraint
from sred.models.base import TimestampMixin, ProvenanceMixin

class StagingStatus(str, Enum):
    PENDING = "PENDING"
    PROMOTED = "PROMOTED"
    ERROR = "ERROR"

class StagingRowType(str, Enum):
    UNKNOWN = "UNKNOWN"
    TIMESHEET = "TIMESHEET"
    PAYROLL = "PAYROLL"
    INVOICE = "INVOICE"
    JIRA = "JIRA"

class StagingRow(TimestampMixin, ProvenanceMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    
    raw_data: str # JSON serialization of row dict
    status: StagingStatus = Field(default=StagingStatus.PENDING)
    
    row_type: StagingRowType = Field(default=StagingRowType.UNKNOWN)
    row_hash: str
    normalized_text: str # For FTS/Embedding
    
    # Source file ID from ProvenanceMixin links back to original CSV

class LedgerLabourHour(TimestampMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    
    person_id: Optional[int] = Field(default=None, foreign_key="person.id")
    
    date: date 
    hours: float
    description: Optional[str] = None
    
    bucket: str = Field(default="UNSORTED")
    inclusion_fraction: float = Field(default=1.0)
    confidence: Optional[float] = None


class PayrollExtract(TimestampMixin, table=True):
    """Structured payroll data extracted from vision artifacts via LLM.

    Each row represents one pay-period total from a payroll document.
    Unique constraint on (run_id, file_id, period_start, period_end) prevents
    duplicate extractions for the same period from the same file.
    """
    __table_args__ = (
        UniqueConstraint("run_id", "file_id", "period_start", "period_end",
                         name="uq_payroll_extract_period"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    file_id: int = Field(foreign_key="file.id")

    period_start: date
    period_end: date
    total_hours: Optional[float] = Field(default=None, description="Total hours if present in payroll")
    total_wages: Optional[float] = Field(default=None, description="Total wages/salary if present")
    currency: str = Field(default="CAD")
    employee_count: Optional[int] = Field(default=None)
    confidence: float = Field(default=0.0, description="LLM extraction confidence 0-1")
    raw_json: str = Field(default="{}", description="Full structured LLM response for audit")
