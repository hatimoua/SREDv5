import streamlit as st
from typing import Optional
from sred.db import get_session as db_get_session
from sred.models.core import Run
from sqlmodel import select

def init_session():
    """Initialize session state variables."""
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None

def get_run_id() -> Optional[int]:
    """Get currently selected run ID."""
    return st.session_state.get("run_id")

def set_run_id(run_id: int):
    """Set currently selected run ID."""
    st.session_state["run_id"] = run_id

def get_current_run_name() -> str:
    """Get name of current run, or empty string."""
    run_id = get_run_id()
    if not run_id:
        return ""
        
    # We could cache this, but for now a quick query is safer
    # Actually, let's just use a session context to fetch
    # This might be tricky inside a render loop if we aren't careful about closing sessions
    # For UI, maybe just return ID or store name in session state when selecting
    return st.session_state.get("run_name", f"Run #{run_id}")

def set_run_context(run: Run):
    """Set context from a Run object."""
    st.session_state["run_id"] = run.id
    st.session_state["run_name"] = run.name
