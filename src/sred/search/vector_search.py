import numpy as np
from typing import List, Tuple
from sqlmodel import Session, select
from sred.models.vector import VectorEmbedding
from sred.logging import logger

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query_vec and all rows in matrix.
    query_vec: (d,)
    matrix: (n, d)
    Returns: (n,) scores
    """
    norm_q = np.linalg.norm(query_vec)
    norm_m = np.linalg.norm(matrix, axis=1)
    
    # Avoid div by zero
    norm_product = norm_q * norm_m
    norm_product[norm_product == 0] = 1e-9
    
    dot_products = np.dot(matrix, query_vec)
    return dot_products / norm_product

def search_vectors(
    session: Session, 
    query_embedding: List[float], 
    run_id: int, 
    top_k: int = 20,
    model: str = "text-embedding-3-small"
) -> List[Tuple[VectorEmbedding, float]]:
    """
    Retrieve top_k most similar vectors for a run.
    """
    # 1. Load all vectors for the run (Model filtered)
    # For PoC with <10k rows, fetching all is fast (<100MB RAM).
    embeddings = session.exec(
        select(VectorEmbedding)
        .where(VectorEmbedding.run_id == run_id, VectorEmbedding.model == model)
    ).all()
    
    if not embeddings:
        return []
    
    # 2. Build matrix
    # Extract arrays
    # VectorEmbedding.vector is bytes
    # We need to parse them
    matrix_list = []
    valid_embeddings = []
    
    for emb in embeddings:
        vec = emb.get_vector()
        if vec.shape[0] == len(query_embedding):
            matrix_list.append(vec)
            valid_embeddings.append(emb)
            
    if not matrix_list:
        return []
        
    matrix = np.array(matrix_list)
    query_vec = np.array(query_embedding, dtype=np.float32)
    
    # 3. Compute scores
    scores = batch_cosine_similarity(query_vec, matrix)
    
    # 4. Sort and Top K
    # argsort returns indices of sorted array (ascending)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((valid_embeddings[idx], float(scores[idx])))
        
    return results
