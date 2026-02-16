import streamlit as st
import pandas as pd
import json
from sqlmodel import Session, select
from sred.db import engine, DATA_DIR
from sred.ui.state import get_run_id
from sred.models.core import File
from sred.models.hypothesis import Hypothesis, StagingMappingProposal
from sred.ingest.csv_intel import csv_profile, csv_query, propose_schema_mapping

st.title("CSV Intelligence Lab")

run_id = get_run_id()
if not run_id:
    st.warning("Select a Run first.")
    st.stop()

# --- Select CSV ---
with Session(engine) as session:
    # Filter for CSVs (by extension or mime)
    files = session.exec(select(File).where(File.run_id == run_id)).all()
    csv_files = [f for f in files if f.original_filename.lower().endswith(".csv")]
    
    if not csv_files:
        st.info("No CSV files found in this run.")
        st.stop()
        
    selected_file_name = st.selectbox("Select CSV File", [f.original_filename for f in csv_files])
    selected_file = next(f for f in csv_files if f.original_filename == selected_file_name)
    file_path = str(DATA_DIR / selected_file.path)

    # --- Labs ---
    tab1, tab2, tab3 = st.tabs(["Profile", "SQL Console", "Schema Hypotheses"])
    
    with tab1:
        st.subheader("File Profile")
        if st.button("Generate Profile", key="btn_profile"):
            try:
                profile = csv_profile(file_path)
                st.metric("Row Count", profile["row_count"])
                
                st.write("**Columns**")
                st.dataframe(profile["columns"])
                
                st.write("**Sample Data**")
                st.dataframe(profile["sample_rows"])
            except Exception as e:
                st.error(f"Profiling failed: {e}")
                
    with tab2:
        st.subheader("DuckDB SQL Console")
        st.caption("Table name is `raw_csv` or creating view `df` implied. Actually the tool handles file reading.")
        st.caption("You can write standard SQL. The file is auto-loaded.")
        
        default_query = f"SELECT * FROM 'read_csv_auto' LIMIT 10" 
        # Actually our wrapper takes raw SQL but injects the table?
        # `csv_query` in csv_intel.py:
        # `con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")`
        # `result = con.execute(sql_query).fetchall()`
        # So user should query `df`.
        
        query = st.text_area("SQL Query", value="SELECT * FROM df LIMIT 5", height=150)
        
        if st.button("Run Query", key="btn_query"):
            try:
                # pass prompt
                # Note: csv_intel.csv_query needs to be robust if we just return rows
                # Let's fix csv_intel.csv_query to return list of dicts -> dataframe
                results = csv_query(file_path, query)
                if isinstance(results, list) and results and "error" in results[0]:
                    st.error(results[0]["error"])
                else:
                    st.dataframe(results)
            except Exception as e:
                st.error(f"Query failed: {e}")

    with tab3:
        st.subheader("Schema Mapping Proposals")
        
        # Check existing
        # Hypothesis linked to run? Or file?
        # StagingMappingProposal linked to file.
        proposals = session.exec(
            select(StagingMappingProposal).where(StagingMappingProposal.file_id == selected_file.id)
        ).all()
        
        if proposals:
            st.success(f"Found {len(proposals)} existing proposals.")
            for p in proposals:
                with st.expander(f"Proposal (Conf: {p.confidence})"):
                    st.json(p.mapping_json)
                    st.write(f"**Reasoning:** {p.reasoning}")
        else:
            st.info("No proposals yet.")
            if st.button("Generate Propopal (LLM)", key="btn_prop"):
                with st.spinner("Analyzing schema..."):
                    propose_schema_mapping(session, selected_file) # This commits
                    st.rerun()
