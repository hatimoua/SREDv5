from sqlalchemy import text
from sqlmodel import Session
from sred.db import engine
from sred.models.core import Segment
from sred.models.memory import MemoryDoc
from sred.logging import logger

def setup_fts():
    """Create FTS5 virtual tables if they don't exist."""
    with Session(engine) as session:
        # Segment FTS
        session.exec(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5(
                id UNINDEXED,
                content,
                content='segment',
                content_rowid='id'
            );
        """))
        
        # MemoryDoc FTS
        session.exec(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                id UNINDEXED,
                content_md,
                content='memorydoc',
                content_rowid='id'
            );
        """))
        
        # Triggers to keep FTS updated are complex to maintain in code.
        # For this PoC, we will rely on explicit reindexing or application-side writes.
        # However, a 'rebuild' command is safer.
        session.commit()

def reindex_all():
    """Rebuild FTS index from source tables."""
    logger.info("Reindexing FTS5 tables...")
    with Session(engine) as session:
        # Clear existing
        session.exec(text("DELETE FROM segment_fts;"))
        session.exec(text("DELETE FROM memory_fts;"))
        
        # Re-insert Segments
        session.exec(text("""
            INSERT INTO segment_fts(rowid, id, content)
            SELECT id, id, content FROM segment;
        """))
        
        # Re-insert MemoryDocs
        session.exec(text("""
            INSERT INTO memory_fts(rowid, id, content_md)
            SELECT id, id, content_md FROM memorydoc;
        """))
        
        session.commit()
    logger.info("Reindexing complete.")

def search_segments(query: str, limit: int = 10):
    with Session(engine) as session:
        results = session.exec(text(f"""
            SELECT id, snippet(segment_fts, 1, '<b>', '</b>', '...', 64) 
            FROM segment_fts 
            WHERE segment_fts MATCH :query 
            ORDER BY rank 
            LIMIT :limit
        """), params={"query": query, "limit": limit}).all()
        return results
