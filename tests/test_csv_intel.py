import pytest
import os
import json
from unittest.mock import MagicMock, patch
from sred.ingest.csv_intel import csv_profile, csv_query, propose_schema_mapping
from sred.models.core import File, Run
from sred.models.hypothesis import Hypothesis, StagingMappingProposal, HypothesisType
from sqlmodel import Session, SQLModel, create_engine, select
from pathlib import Path

@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,hours\n1,Alice,8.0\n2,Bob,7.5\n")
    return str(csv_path)

def test_profile(sample_csv):
    profile = csv_profile(sample_csv)
    assert profile["row_count"] == 2
    cols = {c["name"] for c in profile["columns"]}
    assert "hours" in cols
    assert len(profile["sample_rows"]) == 2

def test_query(sample_csv):
    # Query must use 'df' view
    # But csv_query implementation creates view 'df' from file_path
    # So user query should proceed 'FROM df'
    rows = csv_query(sample_csv, "SELECT * FROM df WHERE hours > 7.9")
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

def test_proposal(session):
    run = Run(name="Test")
    session.add(run)
    session.commit()
    
    file = File(run_id=run.id, path="dummy.csv", original_filename="dummy.csv", mime_type="text/csv", size_bytes=10, content_hash="abc", file_type="text/csv")
    session.add(file)
    session.commit()
    
    # Mock profile and LLM
    with patch("sred.ingest.csv_intel.csv_profile") as mock_prof:
        mock_prof.return_value = {
            "columns": [{"name": "hours", "type": "FLOAT"}], 
            "sample_rows": [],
            "row_count": 10
        }
        
        with patch("sred.ingest.csv_intel.get_chat_completion") as mock_llm:
            mock_llm.return_value = json.dumps({
                "mappings": [
                    {
                        "heuristic_name": "Test Map",
                        "column_map": {"hours": "hours"},
                        "confidence": 0.9,
                        "reasoning": "Obvious match"
                    }
                ]
            })
            
            with patch("sred.ingest.csv_intel.DATA_DIR", new=MagicMock()):
                propose_schema_mapping(session, file)
            
    # Check DB
    hyp = session.exec(select(Hypothesis)).first()
    assert hyp.type == HypothesisType.CSV_SCHEMA
    
    prop = session.exec(select(StagingMappingProposal)).first()
    assert prop.confidence == 0.9
    assert prop.mapping["hours"] == "hours"
