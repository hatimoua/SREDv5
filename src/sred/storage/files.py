import hashlib
import re
from pathlib import Path
from typing import Tuple
from sred.db import DATA_DIR
from sred.logging import logger

def sanitize_filename(name: str) -> str:
    """Sanitize filename to be safe for filesystem."""
    # Remove non-alphanumeric (except ._-)
    s = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    return s.strip('_')

def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()

def save_upload(run_id: int, uploaded_file) -> Tuple[str, str, int, str]:
    """
    Save uploaded file to local storage.
    
    Args:
        run_id: ID of the run context.
        uploaded_file: Streamlit UploadedFile object (or similar with .read(), .name, .type).
        
    Returns:
        Tuple containing:
        - stored_path (str): Relative path to DATA_DIR.
        - sha256 (str): SHA256 hash of content.
        - size_bytes (int): Size in bytes.
        - mime_type (str): MIME type.
    """
    # Create upload directory for run if not exists
    run_dir = DATA_DIR / "runs" / str(run_id) / "uploads"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Read content
    content = uploaded_file.getvalue()
    sha256 = compute_sha256(content)
    size_bytes = len(content)
    mime_type = uploaded_file.type
    original_filename = uploaded_file.name
    
    # Generate stored filename with hash to avoid collisions (though user asked for dedupe logic at DB level, 
    # storing with hash prefix is cleaner).
    safe_name = sanitize_filename(original_filename)
    stored_filename = f"{sha256}_{safe_name}"
    stored_path_abs = run_dir / stored_filename
    
    # Write to disk
    # We might overwrite if same file uploaded for same run, which is fine/desired for idempotency.
    with open(stored_path_abs, "wb") as f:
        f.write(content)
        
    # Return relative path for portability
    stored_path_rel = str(stored_path_abs.relative_to(DATA_DIR))
    
    return stored_path_rel, sha256, size_bytes, mime_type
