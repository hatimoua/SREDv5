import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from pdf2image import convert_from_path
from sred.llm.openai_client import get_vision_completion
from sred.logging import logger

VISION_PROMPT = """
Extract all text from this page exactly as it appears. 
Preserve the layout structure where possible using Markdown.
Identify any tables and format them as Markdown tables.
If there are diagrams or charts, describe them briefly in [brackets].
Output ONLY the markdown content, no preamble.
"""

def pdf_to_images(pdf_path: str) -> List[tuple[int, str]]:
    """
    Convert PDF to a list of (page_number, image_path).
    Uses a temporary directory for images.
    """
    try:
        # We need a temp dir that persists long enough for processing
        # But we want to clean it up. 
        # For simplicity in this PoC, we'll let the user manage temp files or use a context manager in orchestrator.
        # Here we just generate them.
        # Actually pdf2image can return PIL images directly. We can save them to temp files.
        images = convert_from_path(pdf_path)
        
        image_paths = []
        temp_dir = Path(tempfile.mkdtemp(prefix="sred_vision_"))
        
        for i, img in enumerate(images, start=1):
            img_path = temp_dir / f"page_{i}.jpg"
            img.save(img_path, "JPEG")
            image_paths.append((i, str(img_path)))
            
        return image_paths
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise

def vision_extract_image(image_path: str) -> str:
    """
    Extract text from an image using OpenAI Vision.
    """
    return get_vision_completion(image_path, VISION_PROMPT)

def vision_extract_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Process a PDF file.
    Returns list of dicts:
    [
        {"page_number": 1, "text": "...", "confidence": 1.0},
        ...
    ]
    """
    # 1. Convert to images
    page_images = pdf_to_images(pdf_path)
    
    results = []
    total_pages = len(page_images)
    logger.info(f"Extracting {total_pages} pages from PDF...")
    
    # 2. Process each page
    for page_num, img_path in page_images:
        logger.info(f"Processing page {page_num}/{total_pages}...")
        text = vision_extract_image(img_path)
        
        results.append({
            "page_number": page_num,
            "text": text,
            "confidence": 1.0 # Placeholder, model doesn't give confidence easily without logprobs (not supported in vision yet?)
        })
        
    return results
