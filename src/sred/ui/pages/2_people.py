import streamlit as st
from sqlmodel import Session, select
from sred.db import engine
from sred.models.core import Person, Run, RateStatus
from sred.ui.state import get_run_id

st.title("People & Roles")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()
    
with Session(engine) as session:
    # --- Add Person ---
    with st.expander("Add Person", expanded=False):
        with st.form("add_person"):
            name = st.text_input("Name (Required)")
            role = st.text_input("Role (Required)")
            rate = st.number_input("Hourly Rate ($)", min_value=0.0, step=1.0, value=0.0)
            
            submitted = st.form_submit_button("Add Person")
            if submitted:
                if not name or not role:
                    st.error("Name and Role are required.")
                else:
                    status = RateStatus.SET if rate > 0 else RateStatus.PENDING
                    # Treat 0.0 as pending if user didn't change it effectively, or explicit 0 is allowed but handled as set?
                    # Prompt says: "rate_status auto: SET if hourly_rate provided, PENDING otherwise"
                    # In UI providing 0 usually means not provided. Let's assume > 0 is provided.
                    
                    person = Person(
                        run_id=run_id,
                        name=name,
                        role=role,
                        hourly_rate=rate if rate > 0 else None,
                        rate_status=status
                    )
                    session.add(person)
                    session.commit()
                    st.success(f"Added {name}.")
                    st.rerun()

    st.divider()

    # --- List People ---
    people = session.exec(select(Person).where(Person.run_id == run_id)).all()
    
    # Check for warnings
    pending_rates = [p for p in people if p.rate_status == RateStatus.PENDING]
    if pending_rates:
        st.warning(f"⚠️ {len(pending_rates)} people have PENDING rates. This will block claim generation.")
    
    if not people:
        st.info("No people added yet.")
    else:
        for p in people:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
                c1.write(f"**{p.name}**")
                c2.write(f"_{p.role}_")
                
                rate_str = f"${p.hourly_rate}" if p.hourly_rate else "Pending"
                c3.write(f"Rate: {rate_str}")
                
                # Inline edit for rate
                if p.rate_status == RateStatus.PENDING:
                    new_rate = c4.number_input("Set Rate", min_value=0.0, key=f"rate_{p.id}")
                    if c4.button("Save", key=f"save_{p.id}"):
                        if new_rate > 0:
                            p.hourly_rate = new_rate
                            p.rate_status = RateStatus.SET
                            session.add(p)
                            session.commit()
                            st.rerun()
                else:
                    c4.success("Rate Set")
