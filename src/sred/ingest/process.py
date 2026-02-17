import json
import pandas as pd
from pathlib import Path
from sqlmodel import Session, select
from sred.db import engine, DATA_DIR
from sred.models.core import File, FileStatus, Segment, SegmentStatus
from sred.models.artifact import ExtractionArtifact, ArtifactKind
from sred.models.finance import StagingRow, StagingStatus
from sred.ingest.vision import vision_extract_pdf, vision_extract_image
from sred.ingest.segment import create_text_segments, process_csv_content, chunk_text
from sred.search.fts import index_segments
from sred.logging import logger
import docx

def process_source_file(file_id: int):
    """
    Process a source file (PDF, Image, CSV, Text, DOCX).
    - Extracts text/tables.
    - Creates ExtractionArtifacts.
    - Creates Segments/StagingRows.
    - Updates File status.
    """
    with Session(engine) as session:
        file = session.get(File, file_id)
        if not file:
            logger.error(f"File {file_id} not found.")
            return
            
        try:
            # 1. Check if already processed (Artifacts exist for this hash)
            # Actually, we check if File status is PROCESSED first? 
            # Or if artifacts exist for this content_hash in generic way (maybe from another run?)
            # Prompt says: "If SourceFile.sha256 already has VISION_TEXT artifacts for this run, do not re-call OpenAI."
            # So let's check artifacts for this run + hash.
            # But artifacts are linked to File, so we check if ANY file with same hash in this run has artifacts?
            # Or just check if THIS file has artifacts.
            # If we uploaded duplicates, we might have multiple File records with same hash.
            # Let's check for artifacts linked to any file with same hash in same run.
            
            existing_artifacts = session.exec(
                select(ExtractionArtifact)
                .join(File, ExtractionArtifact.file_id == File.id) # Explicit join condition
                .where(File.content_hash == file.content_hash, File.run_id == file.run_id, ExtractionArtifact.kind == ArtifactKind.VISION_TEXT)
            ).first()
            
            # Absolute path
            file_path = DATA_DIR / file.path
            
            if existing_artifacts:
                logger.info(f"Using cached artifacts for {file.original_filename}")
                # We still need to create Segments for THIS file instance if they don't exist.
                # But we skip Vision API call.
                # We can reuse the content from the artifact to create segments.
                # Let's retrieve all text artifacts for this hash.
                cached_texts = session.exec(
                    select(ExtractionArtifact)
                    .join(File, ExtractionArtifact.file_id == File.id)
                    .where(File.content_hash == file.content_hash, File.run_id == file.run_id, ExtractionArtifact.kind == ArtifactKind.VISION_TEXT)
                ).all()
                
                # Re-create segments from these artifacts?
                # A bit complex because artifacts might be page-based.
                # If we have cached artifacts, we should use their data.
                
                all_segs = []
                for art in cached_texts:
                    # art.data contains the text
                    # We can create segments from it.
                    # But we need to handle pagination if art has it. 
                    # art.data is string. 
                    # If it's VISION_TEXT, it's just the text.
                    # We might have stored page info in segment_ids? No, segment_ids is backlink.
                    # ProvenanceMixin has page_number.
                    all_segs.extend(create_text_segments(session, file, art.data, page_number=art.page_number))
                
                file.status = FileStatus.PROCESSED
                session.add(file)
                session.commit()
                index_segments([s.id for s in all_segs if s.id])
                return

            # 2. Process based on type
            mime = file.mime_type.lower()
            ext = Path(file.original_filename).suffix.lower()
            
            if "pdf" in mime or ext == ".pdf":
                logger.info(f"Processing PDF: {file.original_filename}")
                pages = vision_extract_pdf(str(file_path))
                
                for page in pages:
                    text = page["text"]
                    page_num = page["page_number"]
                    
                    # Store Artifact
                    art = ExtractionArtifact(
                        file_id=file.id,
                        run_id=file.run_id,
                        kind=ArtifactKind.VISION_TEXT,
                        data=text,
                        model="gpt-4o", # Assume default for now or pass from config
                        confidence=page.get("confidence"),
                        page_number=page_num
                    )
                    session.add(art)
                    session.commit() # Commit artifact to get ID
                    
                    # Create Segments
                    segs = create_text_segments(session, file, text, page_number=page_num)
                    session.commit() # Commit segments to get IDs
                    
                    # Link Artifact to Segments
                    art.segment_ids_json = json.dumps([s.id for s in segs])
                    session.add(art)
                    session.commit()
                    
            elif "image" in mime or ext in [".jpg", ".jpeg", ".png"]:
                logger.info(f"Processing Image: {file.original_filename}")
                text = vision_extract_image(str(file_path))
                
                art = ExtractionArtifact(
                    file_id=file.id,
                    run_id=file.run_id,
                    kind=ArtifactKind.VISION_TEXT,
                    data=text,
                    model="gpt-4o"
                )
                session.add(art)
                session.commit()
                
                segs = create_text_segments(session, file, text)
                session.commit()
                
                art.segment_ids_json = json.dumps([s.id for s in segs])
                session.add(art)
                session.commit()

            elif "csv" in mime or ext == ".csv":
                logger.info(f"Processing CSV: {file.original_filename}")
                df = pd.read_csv(file_path)
                process_csv_content(session, file, df) # Creates rows and segments
                session.commit()
                
            elif "officedocument.wordprocessingml.document" in mime or ext == ".docx":
                logger.info(f"Processing DOCX: {file.original_filename}")
                doc = docx.Document(file_path)
                full_text = "\n".join([p.text for p in doc.paragraphs])
                create_text_segments(session, file, full_text)
                session.commit()
                
            elif "text" in mime or ext in [".txt", ".md", ".json"]:
                logger.info(f"Processing Text: {file.original_filename}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                create_text_segments(session, file, content)
                session.commit()
                
            else:
                logger.warning(f"Unsupported file type: {mime} / {ext}")
                # Treat as text fallback? Or error?
                # Let's error for now or just skip
                file.status = FileStatus.ERROR
                session.add(file)
                session.commit()
                return

            file.status = FileStatus.PROCESSED
            session.add(file)
            session.commit()

            # Incrementally index all new segments into FTS
            new_seg_ids = [s.id for s in session.exec(
                select(Segment).where(Segment.file_id == file.id)
            ).all() if s.id]
            index_segments(new_seg_ids)

            logger.info(f"Successfully processed {file.original_filename}")
            
        except Exception as e:
            logger.error(f"Failed to process file {file_id}: {e}")
            file.status = FileStatus.ERROR
            session.add(file)
            session.commit()
            raise e
