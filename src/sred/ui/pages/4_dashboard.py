import streamlit as st
from sqlmodel import Session, select, func
from sred.db import engine
from sred.models.core import Run, Person, File, RateStatus
from sred.ui.state import get_run_id

st.title("Run Dashboard")

run_id = get_run_id()

if not run_id:
    st.warning("Please select a Run to view status.")
    st.stop()

with Session(engine) as session:
    run = session.get(Run, run_id)
    
    # Stats
    person_count = session.exec(select(func.count(Person.id)).where(Person.run_id == run_id)).one()
    pending_rates = session.exec(select(func.count(Person.id)).where(Person.run_id == run_id, Person.rate_status == RateStatus.PENDING)).one()
    file_count = session.exec(select(func.count(File.id)).where(File.run_id == run_id)).one()
    
    # --- Status Cards ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Run Status", run.status)
    c2.metric("People", person_count, delta=f"-{pending_rates} Pending" if pending_rates else "All Set")
    c3.metric("Files", file_count)
    
    st.divider()
    
    # --- Readiness Checks ---
    st.subheader("Readiness Checklist")
    
    ready = True
    
    if person_count == 0:
        st.error("❌ No people added. Go to 'People' page.")
        ready = False
    elif pending_rates > 0:
        st.warning(f"⚠️ {pending_rates} people have missing rates. Claim generation will be blocked.")
        # Not a hard block for viewing, but huge warning
        ready = False # Conceptual block
    else:
        st.success("✅ People data complete.")
        
    if file_count == 0:
        st.error("❌ No files uploaded. Go to 'Uploads' page.")
        ready = False
    else:
        st.success("✅ Files uploaded.")
        
    # Placeholder for tasks
    st.info("ℹ️ Blocking Tasks: None (Task system not implemented yet)")
    
    st.divider()
    
    if ready:
        st.success("🚀 Ready for Processing (Coming Soon)")
    else:
        st.warning("Please resolve issues above to proceed.")
