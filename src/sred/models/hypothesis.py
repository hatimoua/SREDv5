from enum import Enum
from typing import Optional, List, Dict
from sqlmodel import Field, SQLModel, Relationship
from datetime import datetime
from sred.models.base import TimestampMixin
import json

class HypothesisType(str, Enum):
    CSV_SCHEMA = "CSV_SCHEMA"
    CLAIM_CLUSTERING = "CLAIM_CLUSTERING"

class HypothesisStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"

class Hypothesis(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    
    type: HypothesisType
    description: str
    status: HypothesisStatus = Field(default=HypothesisStatus.ACTIVE)
    
    parent_id: Optional[int] = Field(default=None, foreign_key="hypothesis.id")
    
    # Relationships
    mapping_proposals: List["StagingMappingProposal"] = Relationship(back_populates="hypothesis")


class StagingMappingProposal(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    
    hypothesis_id: int = Field(foreign_key="hypothesis.id")
    file_id: int = Field(foreign_key="file.id")
    
    mapping_json: str # JSON string of Dict[str, str] (CSV Col -> Target Field)
    confidence: float
    reasoning: str
    
    hypothesis: Optional[Hypothesis] = Relationship(back_populates="mapping_proposals")
    
    @property
    def mapping(self) -> Dict[str, str]:
        return json.loads(self.mapping_json)
    
    @mapping.setter
    def mapping(self, value: Dict[str, str]):
        self.mapping_json = json.dumps(value)
