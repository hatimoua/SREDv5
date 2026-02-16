import pytest
from unittest.mock import MagicMock, patch
from sred.ingest.segment import chunk_text
from sred.ingest.process import process_source_file
from sred.models.core import File, Run, FileStatus, Segment
from sred.models.artifact import ExtractionArtifact, ArtifactKind
from sqlmodel import Session, SQLModel, create_engine, select
from sred.logging import logger

@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

def test_chunking():
    text = "Para 1.\n\nPara 2."
    # With max_chars=100, they fit in one.
    chunks = chunk_text(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == "Para 1.\n\nPara 2."
    
    # Force split
    chunks = chunk_text(text, max_chars=10) # "Para 1." is 7 chars. 
    # "Para 1." fits.
    # "Para 2." fits new chunk.
    assert len(chunks) == 2
    assert chunks[0] == "Para 1."
    
    long_text = "a" * 150
    chunks = chunk_text(long_text, max_chars=100)
    assert len(chunks) == 2
    assert len(chunks[0]) == 100

def test_process_source_file_text(session):
    # Setup
    run = Run(name="Test Run")
    session.add(run)
    session.commit()
    
    file = File(
        run_id=run.id,
        path="test.txt",
        original_filename="test.txt",
        mime_type="text/plain",
        size_bytes=10,
        content_hash="abc",
        file_type="text/plain"
    )
    session.add(file)
    session.commit()
    
    # Mock open
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "Hello World"
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        
        # Also need to patch Session in process.py effectively since it creates its own session usually?
        # process_source_file creates its own session context: `with Session(engine) as session:`
        # This makes testing with in-memory DB hard because `process_source_file` uses `sred.db.engine`.
        # We need to override `sred.db.engine` or patch `Session`.
        
        with patch("sred.ingest.process.Session") as mock_session_cls:
            mock_session_cls.return_value.__enter__.return_value = session
            
            # Since path assumes DATA_DIR, we mock that too or file access
            with patch("sred.ingest.process.DATA_DIR", new=MagicMock()) as mock_data_dir:
                mock_data_dir.__truediv__.return_value = "test.txt"
                
                process_source_file(file.id)
                
    session.refresh(file)
    assert file.status == FileStatus.PROCESSED
    
    segs = session.exec(select(Segment).where(Segment.file_id == file.id)).all()
    assert len(segs) == 1
    assert segs[0].content == "Hello World"

def test_process_source_file_pdf(session):
    # Setup
    run = Run(name="Test Run")
    session.add(run)
    session.commit()
    
    file = File(
        run_id=run.id,
        path="test.pdf",
        original_filename="test.pdf",
        mime_type="application/pdf",
        size_bytes=10,
        content_hash="pdfhash",
        file_type="application/pdf"
    )
    session.add(file)
    session.commit()
    
    # Mock Vision flow
    with patch("sred.ingest.process.Session") as mock_session_cls:
        mock_session_cls.return_value.__enter__.return_value = session
        with patch("sred.ingest.process.vision_extract_pdf") as mock_vision:
            mock_vision.return_value = [
                {"page_number": 1, "text": "Page 1 Content", "confidence": 0.9}
            ]
            
            with patch("sred.ingest.process.DATA_DIR", new=MagicMock()):
                process_source_file(file.id)
                
    session.refresh(file)
    assert file.status == FileStatus.PROCESSED
    
    artifacts = session.exec(select(ExtractionArtifact).where(ExtractionArtifact.file_id == file.id)).all()
    assert len(artifacts) == 1
    assert artifacts[0].kind == ArtifactKind.VISION_TEXT
    assert artifacts[0].data == "Page 1 Content"
    
    segs = session.exec(select(Segment).where(Segment.file_id == file.id)).all()
    assert len(segs) == 1
    assert segs[0].content == "Page 1 Content"
