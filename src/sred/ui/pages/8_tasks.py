import streamlit as st
from sqlmodel import Session, select, desc
from sred.db import engine
from sred.ui.state import get_run_id
from sred.models.core import Run, RunStatus
from sred.models.world import (
    Contradiction, ContradictionStatus, ContradictionSeverity,
    ReviewTask, ReviewTaskStatus,
    ReviewDecision, DecisionLock,
)
from sred.gates import update_run_gate_status

st.title("Tasks & Gates")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()

with Session(engine) as session:
    run = session.get(Run, run_id)

    # --- Gate Status Banner ---
    gate_status = update_run_gate_status(session, run_id)
    if gate_status == RunStatus.NEEDS_REVIEW:
        st.error("**Run is blocked (NEEDS_REVIEW).** Resolve all BLOCKING contradictions and tasks before proceeding.")
    else:
        st.success(f"Run status: **{run.status.value}** — no blocking issues.")

    st.divider()

    # =========================================================================
    # CONTRADICTIONS
    # =========================================================================
    st.subheader("Contradictions")

    contradictions = session.exec(
        select(Contradiction)
        .where(Contradiction.run_id == run_id)
        .order_by(desc(Contradiction.created_at))
    ).all()

    open_contradictions = [c for c in contradictions if c.status == ContradictionStatus.OPEN]
    resolved_contradictions = [c for c in contradictions if c.status == ContradictionStatus.RESOLVED]

    if not contradictions:
        st.info("No contradictions recorded.")
    else:
        st.caption(f"{len(open_contradictions)} open, {len(resolved_contradictions)} resolved")

        for c in contradictions:
            sev_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "BLOCKING": "🔴"}.get(c.severity.value, "⚪")
            status_icon = "🔓" if c.status == ContradictionStatus.RESOLVED else "⚠️"

            with st.expander(f"{status_icon} {sev_icon} [{c.contradiction_type.value}] {c.description[:80]}"):
                st.write(f"**Issue Key:** `{c.issue_key}`")
                st.write(f"**Severity:** {c.severity.value} | **Status:** {c.status.value}")
                st.write(f"**Description:** {c.description}")
                if c.related_entity_type:
                    st.write(f"**Related:** {c.related_entity_type} #{c.related_entity_id}")

    st.divider()

    # =========================================================================
    # REVIEW TASKS
    # =========================================================================
    st.subheader("Review Tasks")

    tasks = session.exec(
        select(ReviewTask)
        .where(ReviewTask.run_id == run_id)
        .order_by(desc(ReviewTask.created_at))
    ).all()

    open_tasks = [t for t in tasks if t.status == ReviewTaskStatus.OPEN]

    if not tasks:
        st.info("No review tasks.")
    else:
        st.caption(f"{len(open_tasks)} open, {len(tasks) - len(open_tasks)} resolved/superseded")

        for t in tasks:
            sev_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "BLOCKING": "🔴"}.get(t.severity.value, "⚪")
            is_open = t.status == ReviewTaskStatus.OPEN

            with st.expander(f"{'📋' if is_open else '✅'} {sev_icon} {t.title} ({t.status.value})"):
                st.write(f"**Issue Key:** `{t.issue_key}`")
                st.write(f"**Severity:** {t.severity.value}")
                st.write(f"**Description:** {t.description}")

                if is_open:
                    st.markdown("---")
                    st.markdown("**Resolve this task:**")
                    decision_text = st.text_area(
                        "Decision / Resolution",
                        key=f"decision_{t.id}",
                        placeholder="Describe how this issue was resolved...",
                    )
                    if st.button("Resolve & Lock", key=f"resolve_{t.id}", type="primary"):
                        if not decision_text.strip():
                            st.warning("Please enter a decision.")
                        else:
                            # 1. Create ReviewDecision
                            decision = ReviewDecision(
                                run_id=run_id,
                                task_id=t.id,
                                decision=decision_text.strip(),
                                decided_by="HUMAN",
                            )
                            session.add(decision)
                            session.commit()
                            session.refresh(decision)

                            # 2. Create DecisionLock
                            lock = DecisionLock(
                                run_id=run_id,
                                issue_key=t.issue_key,
                                decision_id=decision.id,
                                reason=decision_text.strip(),
                                active=True,
                            )
                            session.add(lock)

                            # 3. Mark task resolved
                            t.status = ReviewTaskStatus.RESOLVED
                            session.add(t)

                            # 4. Resolve linked contradiction if any
                            if t.contradiction_id:
                                contradiction = session.get(Contradiction, t.contradiction_id)
                                if contradiction:
                                    contradiction.status = ContradictionStatus.RESOLVED
                                    session.add(contradiction)

                            session.commit()

                            # 5. Re-evaluate gate
                            update_run_gate_status(session, run_id)

                            st.success(f"Task resolved and locked: `{t.issue_key}`")
                            st.rerun()

    st.divider()

    # =========================================================================
    # DECISION LOCKS
    # =========================================================================
    st.subheader("Decision Locks")

    locks = session.exec(
        select(DecisionLock)
        .where(DecisionLock.run_id == run_id)
        .order_by(desc(DecisionLock.created_at))
    ).all()

    active_locks = [lk for lk in locks if lk.active]
    inactive_locks = [lk for lk in locks if not lk.active]

    if not locks:
        st.info("No decision locks.")
    else:
        st.caption(f"{len(active_locks)} active, {len(inactive_locks)} superseded")

        for lk in locks:
            icon = "🔒" if lk.active else "🔓"
            with st.expander(f"{icon} `{lk.issue_key}` — {'ACTIVE' if lk.active else 'SUPERSEDED'}"):
                st.write(f"**Reason:** {lk.reason}")
                st.write(f"**Created:** {lk.created_at.strftime('%Y-%m-%d %H:%M')}")

                if lk.active:
                    st.markdown("---")
                    new_reason = st.text_area(
                        "New decision (supersede)",
                        key=f"supersede_{lk.id}",
                        placeholder="Why are you overriding this lock?",
                    )
                    if st.button("Supersede Lock", key=f"sup_btn_{lk.id}"):
                        if not new_reason.strip():
                            st.warning("Please enter a reason.")
                        else:
                            # 1. Deactivate old lock
                            lk.active = False
                            session.add(lk)

                            # 2. Create new ReviewDecision for the supersede
                            # Find the original task via issue_key
                            orig_task = session.exec(
                                select(ReviewTask).where(
                                    ReviewTask.run_id == run_id,
                                    ReviewTask.issue_key == lk.issue_key,
                                    ReviewTask.status == ReviewTaskStatus.RESOLVED,
                                )
                            ).first()

                            new_decision = ReviewDecision(
                                run_id=run_id,
                                task_id=orig_task.id if orig_task else lk.decision_id,
                                decision=f"[SUPERSEDE] {new_reason.strip()}",
                                decided_by="HUMAN",
                            )
                            session.add(new_decision)
                            session.commit()
                            session.refresh(new_decision)

                            # 3. Create new active lock
                            new_lock = DecisionLock(
                                run_id=run_id,
                                issue_key=lk.issue_key,
                                decision_id=new_decision.id,
                                reason=new_reason.strip(),
                                active=True,
                            )
                            session.add(new_lock)
                            session.commit()

                            # 4. Re-evaluate gate
                            update_run_gate_status(session, run_id)

                            st.success(f"Lock superseded for `{lk.issue_key}`")
                            st.rerun()
