from datetime import date
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel
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
