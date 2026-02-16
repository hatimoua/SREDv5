import streamlit as st
import json
from sqlmodel import Session, select
from sred.db import engine
from sred.ui.state import get_run_id
from sred.models.finance import PayrollExtract, StagingRow, StagingRowType
from sred.models.world import Contradiction, ContradictionType, ContradictionStatus
from sred.config import settings
from datetime import date as date_type

st.title("Payroll Validation")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()

threshold = settings.PAYROLL_MISMATCH_THRESHOLD

# --- Payroll Extracts ---
st.subheader("Payroll Extracts")

with Session(engine) as session:
    extracts = session.exec(
        select(PayrollExtract).where(PayrollExtract.run_id == run_id)
    ).all()

    if not extracts:
        st.info("No payroll extracts yet. Use the Agent to run `payroll_extract` on a payroll file.")
    else:
        for e in extracts:
            with st.expander(
                f"Period: {e.period_start} to {e.period_end} — "
                f"Hours: {e.total_hours or 'N/A'} | Wages: {e.total_wages or 'N/A'} | "
                f"Confidence: {e.confidence:.0%}"
            ):
                cols = st.columns(4)
                cols[0].metric("Hours", f"{e.total_hours or 'N/A'}")
                cols[1].metric("Wages", f"${e.total_wages:,.2f}" if e.total_wages else "N/A")
                cols[2].metric("Employees", e.employee_count or "N/A")
                cols[3].metric("Confidence", f"{e.confidence:.0%}")
                st.caption(f"File ID: {e.file_id} | Currency: {e.currency}")

st.divider()

# --- Mismatch Breakdown ---
st.subheader("Mismatch Breakdown")

with Session(engine) as session:
    extracts = session.exec(
        select(PayrollExtract).where(PayrollExtract.run_id == run_id)
    ).all()

    ts_rows = session.exec(
        select(StagingRow).where(
            StagingRow.run_id == run_id,
            StagingRow.row_type == StagingRowType.TIMESHEET,
        )
    ).all()

    if not extracts:
        st.info("No payroll extracts to compare.")
        st.stop()

    if not ts_rows:
        st.warning("No timesheet data to compare against.")
        st.stop()

    # Sum timesheet hours by date
    ts_hours_by_date: dict[str, float] = {}
    ts_total = 0.0
    for sr in ts_rows:
        try:
            row_dict = json.loads(sr.raw_data)
        except json.JSONDecodeError:
            continue
        h = row_dict.get("hours")
        d = row_dict.get("date")
        if h is not None:
            try:
                hours_val = float(h)
            except (ValueError, TypeError):
                continue
            ts_total += hours_val
            if d:
                ts_hours_by_date[str(d)] = ts_hours_by_date.get(str(d), 0.0) + hours_val

    # Build comparison table
    rows_data = []
    payroll_total = 0.0

    for pe in extracts:
        if pe.total_hours is None:
            rows_data.append({
                "Period": f"{pe.period_start} → {pe.period_end}",
                "Payroll Hours": "N/A",
                "Timesheet Hours": "N/A",
                "Mismatch %": "N/A",
                "Status": "⚪ No hours data",
            })
            continue

        payroll_total += pe.total_hours

        period_ts = 0.0
        for date_str, hrs in ts_hours_by_date.items():
            try:
                d = date_type.fromisoformat(date_str)
            except ValueError:
                continue
            if pe.period_start <= d <= pe.period_end:
                period_ts += hrs

        if pe.total_hours == 0 and period_ts == 0:
            mismatch = 0.0
        elif pe.total_hours == 0:
            mismatch = 1.0
        else:
            mismatch = abs(pe.total_hours - period_ts) / pe.total_hours

        is_blocking = mismatch > threshold

        rows_data.append({
            "Period": f"{pe.period_start} → {pe.period_end}",
            "Payroll Hours": f"{pe.total_hours:.1f}",
            "Timesheet Hours": f"{period_ts:.1f}",
            "Mismatch %": f"{mismatch * 100:.1f}%",
            "Status": "🔴 BLOCKING" if is_blocking else "🟢 OK",
        })

    st.table(rows_data)

    # Overall
    st.divider()
    st.subheader("Overall Summary")

    if payroll_total == 0 and ts_total == 0:
        overall_mismatch = 0.0
    elif payroll_total == 0:
        overall_mismatch = 1.0
    else:
        overall_mismatch = abs(payroll_total - ts_total) / payroll_total

    cols = st.columns(4)
    cols[0].metric("Payroll Total", f"{payroll_total:.1f}h")
    cols[1].metric("Timesheet Total", f"{ts_total:.1f}h")
    cols[2].metric("Mismatch", f"{overall_mismatch * 100:.1f}%")
    cols[3].metric("Threshold", f"{threshold * 100:.0f}%")

    if overall_mismatch > threshold:
        st.error(
            f"Overall mismatch ({overall_mismatch*100:.1f}%) exceeds threshold "
            f"({threshold*100:.0f}%). Check Tasks & Gates for BLOCKING contradictions."
        )
    else:
        st.success(f"Overall mismatch ({overall_mismatch*100:.1f}%) is within threshold.")

    # Show related contradictions
    st.divider()
    st.subheader("Payroll Contradictions")
    contradictions = session.exec(
        select(Contradiction).where(
            Contradiction.run_id == run_id,
            Contradiction.contradiction_type == ContradictionType.PAYROLL_MISMATCH,
        )
    ).all()

    if not contradictions:
        st.info("No payroll mismatch contradictions.")
    else:
        for c in contradictions:
            icon = "🔴" if c.status == ContradictionStatus.OPEN else "✅"
            with st.expander(f"{icon} {c.issue_key} — {c.severity.value} — {c.status.value}"):
                st.write(c.description)
