import streamlit as st
import json
from sqlmodel import Session, select, func
from sred.db import engine
from sred.ui.state import get_run_id
from sred.models.finance import LedgerLabourHour, StagingRow, StagingStatus, StagingRowType
from sred.models.core import Person
from sred.models.alias import PersonAlias, AliasStatus

st.title("Labour Ledger")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()

with Session(engine) as session:

    # ------------------------------------------------------------------
    # 1. Summary Metrics
    # ------------------------------------------------------------------
    st.subheader("Summary")

    ledger_rows = session.exec(
        select(LedgerLabourHour).where(LedgerLabourHour.run_id == run_id)
    ).all()

    staging_total = session.exec(
        select(func.count(StagingRow.id)).where(StagingRow.run_id == run_id)
    ).one()
    staging_promoted = session.exec(
        select(func.count(StagingRow.id)).where(
            StagingRow.run_id == run_id,
            StagingRow.status == StagingStatus.PROMOTED,
        )
    ).one()
    staging_pending = session.exec(
        select(func.count(StagingRow.id)).where(
            StagingRow.run_id == run_id,
            StagingRow.status == StagingStatus.PENDING,
        )
    ).one()

    if not ledger_rows:
        cols = st.columns(3)
        cols[0].metric("Staging Rows", staging_total)
        cols[1].metric("Promoted", staging_promoted)
        cols[2].metric("Pending", staging_pending)
        st.info(
            "No ledger entries yet. Use the Agent to run `ledger_populate` "
            "after confirming person aliases."
        )
        st.stop()

    # Aggregate stats
    total_hours = sum(r.hours for r in ledger_rows)
    sred_rows = [r for r in ledger_rows if r.bucket == "SR&ED"]
    sred_hours = sum(r.hours * r.inclusion_fraction for r in sred_rows)
    person_ids = set(r.person_id for r in ledger_rows if r.person_id)
    avg_confidence = (
        sum(r.confidence for r in ledger_rows if r.confidence is not None)
        / max(len([r for r in ledger_rows if r.confidence is not None]), 1)
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ledger Entries", len(ledger_rows))
    c2.metric("Total Hours", f"{total_hours:,.1f}")
    c3.metric("SR&ED Hours", f"{sred_hours:,.1f}")
    c4.metric("People Mapped", len(person_ids))
    c5.metric("Avg Confidence", f"{avg_confidence:.0%}")

    # Staging row progress bar
    if staging_total > 0:
        progress = staging_promoted / staging_total
        st.progress(progress, text=f"Staging rows: {staging_promoted}/{staging_total} promoted ({progress:.0%})")

    st.divider()

    # ------------------------------------------------------------------
    # 2. Per-Person Breakdown
    # ------------------------------------------------------------------
    st.subheader("Per-Person Breakdown")

    # Build person lookup
    persons = session.exec(
        select(Person).where(Person.run_id == run_id)
    ).all()
    person_map = {p.id: p for p in persons}

    # Group ledger rows by person
    by_person: dict[int, list] = {}
    for r in ledger_rows:
        pid = r.person_id or 0
        by_person.setdefault(pid, []).append(r)

    table_data = []
    for pid, rows in sorted(by_person.items(), key=lambda x: x[0]):
        person = person_map.get(pid)
        name = person.name if person else f"Unknown (ID {pid})"
        role = person.role if person else "—"
        p_hours = sum(r.hours for r in rows)
        p_sred = sum(r.hours * r.inclusion_fraction for r in rows if r.bucket == "SR&ED")
        p_incl = p_sred / p_hours if p_hours > 0 else 0.0
        p_conf = sum(r.confidence for r in rows if r.confidence) / max(len([r for r in rows if r.confidence]), 1)
        dates = sorted(set(str(r.date) for r in rows))
        date_range = f"{dates[0]} → {dates[-1]}" if len(dates) > 1 else dates[0] if dates else "—"
        buckets = sorted(set(r.bucket for r in rows))

        table_data.append({
            "Person": name,
            "Role": role,
            "Total Hours": f"{p_hours:,.1f}",
            "SR&ED Hours": f"{p_sred:,.1f}",
            "Inclusion %": f"{p_incl:.1%}",
            "Confidence": f"{p_conf:.0%}",
            "Bucket": ", ".join(buckets),
            "Date Range": date_range,
        })

    st.table(table_data)

    st.divider()

    # ------------------------------------------------------------------
    # 3. Detailed Ledger Entries
    # ------------------------------------------------------------------
    with st.expander("Detailed Ledger Entries", expanded=False):
        detail_data = []
        for r in ledger_rows:
            person = person_map.get(r.person_id)
            detail_data.append({
                "ID": r.id,
                "Person": person.name if person else f"ID {r.person_id}",
                "Date": str(r.date),
                "Hours": f"{r.hours:,.1f}",
                "Bucket": r.bucket,
                "Inclusion": f"{r.inclusion_fraction:.1%}",
                "Confidence": f"{r.confidence:.0%}" if r.confidence else "—",
                "Description": r.description or "—",
            })
        st.dataframe(detail_data, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # 4. Unmatched Staging Rows
    # ------------------------------------------------------------------
    st.subheader("Unmatched Staging Rows")

    pending_rows = session.exec(
        select(StagingRow).where(
            StagingRow.run_id == run_id,
            StagingRow.status == StagingStatus.PENDING,
        )
    ).all()

    if not pending_rows:
        st.success("All staging rows have been promoted to the ledger.")
    else:
        st.warning(f"{len(pending_rows)} staging row(s) still pending — names may not match any confirmed alias.")

        # Load confirmed aliases for reference
        aliases = session.exec(
            select(PersonAlias).where(
                PersonAlias.run_id == run_id,
                PersonAlias.status == AliasStatus.CONFIRMED,
            )
        ).all()
        confirmed_names = {a.alias.strip().lower() for a in aliases}

        unmatched_data = []
        for sr in pending_rows:
            try:
                row = json.loads(sr.raw_data)
            except json.JSONDecodeError:
                row = {}

            # Try common name columns
            name = None
            for col in ["Employee", "employee", "Name", "name", "Full Name"]:
                if col in row:
                    name = str(row[col]).strip()
                    break

            in_aliases = name.lower() in confirmed_names if name else False

            unmatched_data.append({
                "Staging ID": sr.id,
                "Name": name or "—",
                "Type": sr.row_type.value,
                "Has Alias?": "✅" if in_aliases else "❌",
                "Status": sr.status.value,
            })

        st.table(unmatched_data)

        if any(not d["Has Alias?"] == "✅" for d in unmatched_data):
            st.caption(
                "💡 Names without a confirmed alias need to be resolved. "
                "Use the Agent with `aliases_resolve` or `aliases_confirm` to map them to Person records, "
                "then run `ledger_populate` again."
            )
