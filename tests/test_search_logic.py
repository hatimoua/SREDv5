import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sred.search.embeddings import compute_text_hash, store_embeddings
from sred.search.vector_search import cosine_similarity, search_vectors
from sred.search.hybrid_search import rrf_fusion, SearchResult, EntityType
from sred.models.core import Run
from sred.models.vector import VectorEmbedding
from sqlmodel import Session, SQLModel, create_engine
from sred.logging import logger

@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

def test_hash():
    assert compute_text_hash("test") == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

def test_cosine():
    v1 = np.array([1, 0], dtype=np.float32)
    v2 = np.array([1, 0], dtype=np.float32)
    assert cosine_similarity(v1, v2) > 0.99
    
    v3 = np.array([0, 1], dtype=np.float32)
    assert cosine_similarity(v1, v3) < 0.01

def test_vector_storage(session):
    run = Run(name="Test")
    session.add(run)
    session.commit()
    
    # Mock OpenAI
    with patch("sred.search.embeddings.get_embeddings_from_openai") as mock_openai:
        mock_openai.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        texts = ["A", "B"]
        ids = [1, 2]
        store_embeddings(session, texts, ids, EntityType.SEGMENT, run.id)
        
        # Check DB
        vecs = session.query(VectorEmbedding).all()
        assert len(vecs) == 2
        
        # Check caching (should not call openai again)
        mock_openai.reset_mock()
        store_embeddings(session, texts, ids, EntityType.SEGMENT, run.id)
        mock_openai.assert_not_called()

def test_rrf():
    # Setup hits
    fts = [SearchResult(id=1, content="A", score=0, source="FTS", rank_fts=1)]
    vec = [SearchResult(id=1, content="A", score=0.9, source="VEC", rank_vector=1)]
    
    fused = rrf_fusion(fts, vec, k=1)
    # Score for ID 1: 1/(1+1) + 1/(1+1) = 0.5 + 0.5 = 1.0
    assert len(fused) == 1
    assert fused[0].id == 1
    assert abs(fused[0].score - 1.0) < 0.001
