import pandas as pd
from typing import List, Dict, Any, Optional
from sred.models.core import Segment, SegmentStatus, File
from sred.models.finance import StagingRow, StagingStatus
from sqlmodel import Session

MAX_CHUNK_SIZE = 1000

def chunk_text(text: str, max_chars: int = MAX_CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks.
    Tries to split by paragraphs (double newline) first.
    If a paragraph is too long, splits by sentence (naive).
    This is a basic implementation for PoC.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself is huge, force split
            if len(para) > max_chars:
                # Naive chunking by size
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i+max_chars])
                current_chunk = ""
            else:
                current_chunk = para
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def create_text_segments(session: Session, file: File, text: str, page_number: Optional[int] = None) -> List[Segment]:
    """
    Create Segment objects from text content.
    Returns list of created Segment objects (not committed yet).
    """
    chunks = chunk_text(text)
    segments = []
    
    for chunk in chunks:
        seg = Segment(
            file_id=file.id,
            run_id=file.run_id,
            content=chunk,
            page_number=page_number,
            status=SegmentStatus.PENDING
        )
        session.add(seg)
        segments.append(seg)
        
    return segments

def process_csv_content(session: Session, file: File, df: pd.DataFrame) -> List[StagingRow]:
    """
    Process DataFrame rows into StagingRows and Segments.
    """
    rows = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        # Create StagingRow
        staging = StagingRow(
            run_id=file.run_id,
            source_file_id=file.id,
            row_number=index + 1,
            raw_data=pd.json_normalize(row_dict).to_json(orient='records')[1:-1], # Hacky json dump or just json.dumps
            status=StagingStatus.PENDING,
            row_hash=str(hash(frozenset(row_dict.items()))), # Naive hash
            normalized_text=" ".join(str(v) for v in row_dict.values())
        )
        # Fix json dump properly
        import json
        staging.raw_data = json.dumps(row_dict, default=str)
        session.add(staging)
        rows.append(staging)
        
        # Also create a Segment for searchability? 
        # Prompt says: "create StagingRow per row + Segment per row"
        seg = Segment(
            file_id=file.id,
            run_id=file.run_id,
            content=staging.normalized_text,
            row_number=index + 1,
            status=SegmentStatus.PENDING
        )
        session.add(seg)
        
    return rows
