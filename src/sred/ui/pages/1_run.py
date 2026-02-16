import streamlit as st
from sqlmodel import Session, select, desc
from sred.db import engine
from sred.models.core import Run, RunStatus
from sred.ui.state import set_run_context

st.title("Projects & Runs")

with Session(engine) as session:
    # --- Create New ---
    st.subheader("Create New Run")
    with st.form("new_run_form"):
        new_name = st.text_input("Run Name", placeholder="e.g. Acme Corp FY2024")
        submitted = st.form_submit_button("Create Run")
        
        if submitted and new_name:
            run = Run(name=new_name, status=RunStatus.INITIALIZING)
            session.add(run)
            session.commit()
            session.refresh(run)
            st.success(f"Created run '{run.name}' (ID: {run.id})")
            # Auto-select
            st.session_state["run_id"] = run.id
            st.rerun()

    st.divider()

    # --- Select Existing ---
    st.subheader("Existing Runs")
    runs = session.exec(select(Run).order_by(desc(Run.created_at))).all()
    
    if not runs:
        st.info("No runs found. Create one above.")
    else:
        # Display as a table with select
        for run in runs:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            col1.write(f"**{run.name}**")
            col2.write(f"_{run.status}_")
            col3.write(f"ID: {run.id}")
            
            # Highlight current selection
            is_selected = st.session_state.get("run_id") == run.id
            btn_label = "Selected" if is_selected else "Select"
            
            if col4.button(btn_label, key=f"sel_{run.id}", disabled=is_selected):
                set_run_context(run) # No set_run_context helper in state.py yet? Oh I just created it.
                st.rerun()
                
    # Show current context
    current_id = st.session_state.get("run_id")
    if current_id:
        st.sidebar.success(f"Active Run: {session.get(Run, current_id).name}")
    else:
        st.sidebar.warning("No Run Selected")
