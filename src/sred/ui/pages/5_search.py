import streamlit as st
from sqlmodel import Session, select
from sred.db import engine
from sred.ui.state import get_run_id
from sred.search.fts import search_segments
from sred.search.embeddings import get_query_embedding
from sred.search.vector_search import search_vectors
from sred.search.hybrid_search import fts_search, vector_search_wrapper, rrf_fusion
from sred.models.core import Segment, File

st.title("Search & Discovery")

run_id = get_run_id()
if not run_id:
    st.warning("Select a Run first.")
    st.stop()
    
# --- Controls ---
c1, c2 = st.columns([3, 1])
query = c1.text_input("Search Query", placeholder="e.g. 'machine learning research'")
mode = c2.selectbox("Mode", ["Hybrid", "FTS Only", "Vector Only"])

if query:
    with Session(engine) as session:
        st.divider()
        with st.spinner(f"Searching ({mode})..."):
            try:
                if mode == "FTS Only":
                    # Call existing FTS
                    # search_segments returns raw tuples (id, snippet).
                    # Let's use our wrapper in hybrid_search for consistency?
                    # But fts_search implementation above was a placeholder using search_segments essentially.
                    results = fts_search(session, query, limit=20)
                elif mode == "Vector Only":
                    results = vector_search_wrapper(session, query, run_id, limit=20)
                else:
                    # Hybrid
                    # Get FTS
                    fts_hits = fts_search(session, query, limit=20)
                    
                    # Get Vector
                    vec_hits = vector_search_wrapper(session, query, run_id, limit=20)
                    
                    results = rrf_fusion(fts_hits, vec_hits)
                    
                st.subheader(f"Results ({len(results)})")
                
                for res in results:
                    # Fetch extra details for display (File Name, Page)
                    seg = session.get(Segment, res.id)
                    if not seg:
                        continue
                        
                    file = session.get(File, seg.file_id)
                    filename = file.original_filename if file else "Unknown"
                    page = f"Page {seg.page_number}" if seg.page_number else f"Row {seg.row_number}" if seg.row_number else ""
                    
                    with st.container(border=True):
                        st.markdown(f"**{filename}** · _{page}_ · Score: `{res.score:.4f}`")
                        st.markdown(res.content, unsafe_allow_html=True) # FTS snippets have HTML
                        st.caption(f"Source: {res.source}")
                        
            except Exception as e:
                st.error(f"Search failed: {e}")
