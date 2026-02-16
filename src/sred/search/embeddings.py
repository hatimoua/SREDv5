import hashlib
from typing import List, Optional
import numpy as np
from sqlmodel import Session, select
from sred.llm.openai_client import client
from sred.config import settings
from sred.models.vector import VectorEmbedding, EntityType
from sred.logging import logger

EMBEDDING_MODEL = "text-embedding-3-small"

def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_embeddings_from_openai(texts: List[str]) -> List[List[float]]:
    """Batch call to OpenAI Embeddings API."""
    try:
        # OpenAI handles batching, but we should be mindful of limits.
        # For PoC, assume reasonable batch size.
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        # Ensure order is preserved
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"OpenAI Embedding API failed: {e}")
        raise

def get_query_embedding(text: str) -> List[float]:
    """Get embedding for a single query string."""
    return get_embeddings_from_openai([text])[0]

def store_embeddings(
    session: Session, 
    texts: List[str], 
    entity_ids: List[int], 
    entity_type: EntityType, 
    run_id: int
):
    """
    Generate and store embeddings for a batch of entities.
    Checks for existing embeddings by (entity_type, entity_id, model) OR text_hash?
    The constraint is (entity_type, entity_id, model).
    We should check if they already exist for this entity.
    """
    if not texts:
        return

    # Identify missing
    missing_indices = []
    missing_texts = []
    
    for i, (text, eid) in enumerate(zip(texts, entity_ids)):
        # Check specific entity existence
        exists = session.exec(
            select(VectorEmbedding)
            .where(
                VectorEmbedding.entity_type == entity_type,
                VectorEmbedding.entity_id == eid,
                VectorEmbedding.model == EMBEDDING_MODEL
            )
        ).first()
        
        if not exists:
            missing_indices.append(i)
            missing_texts.append(text)
    
    if not missing_texts:
        return

    logger.info(f"Generating embeddings for {len(missing_texts)} items...")
    vectors = get_embeddings_from_openai(missing_texts)
    
    for i, vec in zip(missing_indices, vectors):
        text = texts[i]
        eid = entity_ids[i]
        
        emb = VectorEmbedding(
            run_id=run_id,
            entity_type=entity_type,
            entity_id=eid,
            model=EMBEDDING_MODEL,
            text_hash=compute_text_hash(text)
        )
        emb.set_vector(vec)
        session.add(emb)
        
    session.commit()
