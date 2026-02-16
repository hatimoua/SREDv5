import streamlit as st
from sqlmodel import Session, select, desc
from sred.db import engine
from sred.models.core import File, FileStatus
from sred.storage.files import save_upload
from sred.ui.state import get_run_id

st.title("File Uploads")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()

# --- Uploader ---
uploaded_files = st.file_uploader(
    "Upload Documents", 
    accept_multiple_files=True,
    type=["csv", "pdf", "docx", "txt", "md", "json"]
)

if uploaded_files:
    with Session(engine) as session:
        for uf in uploaded_files:
            # Process one by one (in reality we might want a button to confirm process, but stream is okay)
            # Actually, let's just process immediately as per standard Streamlit flow unless form used
            
            # Check dupes based on hash is tricky before reading bytes.
            # save_upload reads bytes.
            # We can't easily check dupe without reading or trusting name/size.
            # Let's read and save, then check DB.
            
            try:
                # 1. Save to disk (computes hash)
                stored_path, sha256, size, mime = save_upload(run_id, uf)
                
                # 2. Check DB
                existing = session.exec(
                    select(File).where(File.run_id == run_id, File.content_hash == sha256)
                ).first()
                
                if existing:
                    st.toast(f"ℹ️ {uf.name} already uploaded.", icon="ℹ️")
                    continue
                
                # 3. Create DB Record
                new_file = File(
                    run_id=run_id,
                    original_filename=uf.name,
                    path=stored_path,
                    mime_type=mime,
                    size_bytes=size,
                    content_hash=sha256,
                    status=FileStatus.UPLOADED,
                    # legacy
                    file_type=mime 
                )
                session.add(new_file)
                session.commit()
                st.toast(f"✅ Uploaded {uf.name}", icon="✅")
                
            except Exception as e:
                st.error(f"Failed to upload {uf.name}: {e}")

from sred.ingest.process import process_source_file
from sred.logging import logger

st.divider()

# --- List Files ---
with Session(engine) as session:
    files = session.exec(
        select(File).where(File.run_id == run_id).order_by(desc(File.created_at))
    ).all()
    
    if not files:
        st.info("No files uploaded.")
    else:
        st.write(f"Total Files: {len(files)}")
        
        # Display as custom grid to allow actions
        for f in files:
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 2])
                c1.write(f"**{f.original_filename}**")
                c2.write(f.mime_type)
                c3.write(f"{round(f.size_bytes / 1024, 1)} KB")
                
                # Status with icon
                status_icon = "✅" if f.status == FileStatus.PROCESSED else "❌" if f.status == FileStatus.ERROR else "⏳"
                c4.write(f"{status_icon} {f.status.value}")
                
                # Action
                if f.status != FileStatus.PROCESSED:
                    if c5.button("Process", key=f"proc_{f.id}"):
                        with st.spinner("Processing..."):
                            try:
                                process_source_file(f.id)
                                st.success("Done!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    c5.success("Processed")
