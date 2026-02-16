import pytest
from datetime import date
from sqlmodel import Session, SQLModel, create_engine, select
from sqlalchemy.exc import IntegrityError
from sred.models.core import Run, File, Segment, Person, RateStatus, SegmentStatus
from sred.models.vector import VectorEmbedding, EntityType
from sred.models.memory import MemoryDoc
from sred.models.artifact import ExtractionArtifact, ArtifactKind
from sred.models.finance import StagingRow, LedgerLabourHour, StagingStatus, StagingRowType
from sred.search import setup_fts, reindex_all, search_segments
import numpy as np
import json

# Use in-memory DB for testing
@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    
    from sqlalchemy import text
    with Session(engine) as session:
        session.exec(text("CREATE VIRTUAL TABLE segment_fts USING fts5(id UNINDEXED, content, content='segment', content_rowid='id');"))
        session.commit()
    
    with Session(engine) as session:
        yield session

def test_core_models(session: Session):
    run = Run(name="Test Run")
    session.add(run)
    session.commit()
    session.refresh(run)
    
    # Person now needs run_id and role
    person = Person(
        run_id=run.id,
        name="Alice",
        role="Developer",
        hourly_rate=100.0,
        rate_status=RateStatus.SET
    )
    session.add(person)
    session.commit()
    
    file = File(
        run_id=run.id, 
        path="foo.txt", 
        file_type="text/plain", # Legacy field 
        mime_type="text/plain", 
        original_filename="foo.txt",
        size_bytes=123,
        content_hash="abc"
    )
    session.add(file)
    session.commit()
    
    segment = Segment(
        file_id=file.id, 
        run_id=run.id, 
        content="This is a test segment.",
        status=SegmentStatus.PENDING
    )
    session.add(segment)
    session.commit()
    
    assert segment.id is not None
    assert segment.file.path == "foo.txt"
    assert segment.run_id == run.id
    assert person.run_id == run.id

def test_finance_models(session: Session):
    run = Run(name="Finance Run")
    session.add(run)
    session.commit()
    
    row_data = {"col1": "val1", "col2": "val2"}
    row = StagingRow(
        run_id=run.id,
        raw_data=json.dumps(row_data),
        status=StagingStatus.PENDING,
        row_type=StagingRowType.TIMESHEET,
        row_hash="hash123",
        normalized_text="val1 val2"
    )
    session.add(row)
    
    hour = LedgerLabourHour(
        run_id=run.id,
        date=date(2023, 1, 1),
        hours=8.0,
        description="Coding",
        bucket="DEV",
        inclusion_fraction=0.8,
        confidence=0.9
    )
    session.add(hour)
    session.commit()
    
    assert row.id is not None
    assert hour.id is not None
    assert hour.bucket == "DEV"

def test_extraction_artifact(session: Session):
    run = Run(name="Artifact Run")
    session.add(run)
    session.commit()
    file = File(
        run_id=run.id, 
        path="art.txt", 
        file_type="text/plain", 
        mime_type="text/plain",
        original_filename="art.txt",
        size_bytes=10,
        content_hash="123"
    )
    session.add(file)
    session.commit()
    
    art = ExtractionArtifact(
        file_id=file.id,
        run_id=run.id,
        kind=ArtifactKind.VISION_TEXT,
        data="Some extracted text",
        model="gpt-4o",
        segment_ids_json="[1, 2]"
    )
    session.add(art)
    session.commit()
    
    assert art.id is not None
    assert art.segment_ids_json == "[1, 2]"

def test_vector_embedding(session: Session):
    run = Run(name="Vector Run")
    session.add(run)
    session.commit()
    
    vec = [0.1, 0.2, 0.3]
    embedding = VectorEmbedding(
        run_id=run.id,
        entity_type=EntityType.SEGMENT,
        entity_id=1,
        model="test-model",
        text_hash="hash",
        dims=0, 
        vector=b"" 
    )
    embedding.set_vector(vec)
    session.add(embedding)
    session.commit()
    
    # Test duplicate constraint
    dup = VectorEmbedding(
        run_id=run.id,
        entity_type=EntityType.SEGMENT,
        entity_id=1,
        model="test-model", # Same model/entity
        text_hash="hash2",
        dims=0,
        vector=b""
    )
    dup.set_vector(vec)
    session.add(dup)
    
    with pytest.raises(IntegrityError):
        session.commit()
    
    session.rollback()
