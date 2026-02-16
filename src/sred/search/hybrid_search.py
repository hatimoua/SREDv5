from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict
from sqlmodel import Session
from sred.models.core import Segment
from sred.search.fts import search_segments
from sred.search.embeddings import get_query_embedding
from sred.search.vector_search import search_vectors
from sred.models.vector import EntityType

@dataclass
class SearchResult:
    id: int # Segment ID
    content: str
    score: float # Relevant score (sim or rank)
    source: str # e.g. "FTS", "VECTOR", "HYBRID"
    rank_fts: int = 1000
    rank_vector: int = 1000

def fts_search(session: Session, query: str, limit: int = 20) -> List[SearchResult]:
    """
    Perform FTS search.
    Returns SearchResult objects with score=1/(rank).
    """
    raw_results = search_segments(query, limit=limit)
    hits = []
    
    # search_segments returns list of (id, snippet).
    # We need to fetch full content? Or just use snippet?
    # For now, let's fetch Segment object to be consistent.
    # Actually search_segments returns list of rows.
    # Let's just assume we get ID and use that.
    
    for i, row in enumerate(raw_results):
        seg_id = row[0]
        snippet = row[1]
        
        # We need provenance?
        hits.append(SearchResult(
            id=seg_id,
            content=snippet,
            score=0, # Will be set by fusion or just rank
            source="FTS",
            rank_fts=i + 1
        ))
    return hits

def vector_search_wrapper(session: Session, query: str, run_id: int, limit: int = 20) -> List[SearchResult]:
    """
    Perform Vector search.
    """
    # 1. Embed query
    query_vec = get_query_embedding(query)
    
    # 2. Search
    vec_results = search_vectors(session, query_vec, run_id, top_k=limit)
    
    hits = []
    for i, (emb, score) in enumerate(vec_results):
        if emb.entity_type == EntityType.SEGMENT:
            # Fetch segment content
            seg = session.get(Segment, emb.entity_id)
            if seg:
                hits.append(SearchResult(
                    id=seg.id,
                    content=seg.content[:200] + "...", # Snippet
                    score=score,
                    source="VECTOR",
                    rank_vector=i + 1
                ))
    return hits

def rrf_fusion(fts_results: List[SearchResult], vector_results: List[SearchResult], k: int = 60) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion.
    Score = 1/(k + rank_fts) + 1/(k + rank_vector)
    """
    scores = defaultdict(float)
    content_map = {}
    
    # Process FTS
    for res in fts_results:
        scores[res.id] += 1 / (k + res.rank_fts)
        content_map[res.id] = res.content
        
    # Process Vector
    for res in vector_results:
        scores[res.id] += 1 / (k + res.rank_vector)
        # Prefer FTS snippet if available (has highlighting)? Or Vector content?
        # FTS snippet has <b> tags. Keep it if present.
        if res.id not in content_map:
            content_map[res.id] = res.content
            
    # Create fused list
    fused = []
    for seg_id, score in scores.items():
        fused.append(SearchResult(
            id=seg_id,
            content=content_map[seg_id],
            score=score,
            source="HYBRID"
        ))
        
    # Sort descending
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused

def hybrid_search(session: Session, query: str, run_id: int, limit: int = 20) -> List[SearchResult]:
    """
    Perform Hybrid Search (FTS + Vector).
    """
    fts_hits = fts_search(session, query, limit=limit)
    vec_hits = vector_search_wrapper(session, query, run_id, limit=limit)
    
    return rrf_fusion(fts_hits, vec_hits)
