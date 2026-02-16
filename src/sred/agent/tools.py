"""
Tool implementations for the agent runner.

Each tool function signature: handler(session, run_id, **kwargs) -> dict

Rules:
- LLM never writes SQL or DB directly.
- Tools enforce provenance, idempotency, caching.
"""
import json
import hashlib
from typing import Dict, Any
from sqlmodel import Session, select
from sred.models.core import File, Segment, Person, Run, RateStatus
from sred.models.finance import StagingRow
from sred.models.memory import MemoryDoc
from sred.models.hypothesis import Hypothesis, StagingMappingProposal, HypothesisStatus
from sred.models.world import (
    Contradiction, ContradictionSeverity, ContradictionType, ContradictionStatus,
    ReviewTask, ReviewTaskStatus, ReviewDecision, DecisionLock,
)
from sred.gates import has_active_lock, update_run_gate_status
from sred.db import DATA_DIR
from sred.logging import logger
from sred.agent.registry import register_tool


# ---------------------------------------------------------------------------
# ingest.process_file
# ---------------------------------------------------------------------------
def _ingest_process_file(session: Session, run_id: int, *, source_file_id: int) -> dict:
    """Trigger processing for a source file (idempotent)."""
    from sred.ingest.process import process_source_file

    file = session.get(File, source_file_id)
    if not file:
        return {"error": f"File {source_file_id} not found."}
    if file.run_id != run_id:
        return {"error": "File does not belong to the active run."}
    if file.status.value == "PROCESSED":
        return {"status": "already_processed", "file_id": source_file_id}

    process_source_file(source_file_id)
    session.refresh(file)
    return {"status": file.status.value, "file_id": source_file_id, "filename": file.original_filename}


register_tool(
    name="ingest_process_file",
    description="Process an uploaded source file (PDF, CSV, DOCX, TXT, image). Extracts text, creates segments. Idempotent.",
    parameters={
        "type": "object",
        "properties": {
            "source_file_id": {"type": "integer", "description": "ID of the File record to process."},
        },
        "required": ["source_file_id"],
    },
    handler=_ingest_process_file,
)


# ---------------------------------------------------------------------------
# search.hybrid
# ---------------------------------------------------------------------------
def _search_hybrid(session: Session, run_id: int, *, query: str, limit: int = 10) -> dict:
    """Run hybrid (FTS + vector) search over segments."""
    from sred.search.hybrid_search import hybrid_search

    results = hybrid_search(session, query, run_id, limit=limit)
    return {
        "count": len(results),
        "results": [
            {"segment_id": r.id, "score": round(r.score, 4), "snippet": r.content[:300]}
            for r in results
        ],
    }


register_tool(
    name="search_hybrid",
    description="Search segments using hybrid FTS + vector search. Returns ranked results with snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "limit": {"type": "integer", "description": "Max results to return (default 10)."},
        },
        "required": ["query"],
    },
    handler=_search_hybrid,
)


# ---------------------------------------------------------------------------
# csv.profile
# ---------------------------------------------------------------------------
def _csv_profile(session: Session, run_id: int, *, file_id: int) -> dict:
    """Profile a CSV file using DuckDB."""
    from sred.ingest.csv_intel import csv_profile

    file = session.get(File, file_id)
    if not file:
        return {"error": f"File {file_id} not found."}
    if file.run_id != run_id:
        return {"error": "File does not belong to the active run."}

    file_path = str(DATA_DIR / file.path)
    try:
        profile = csv_profile(file_path)
        # Truncate sample rows for token efficiency
        profile["sample_rows"] = profile["sample_rows"][:3]
        return profile
    except Exception as e:
        return {"error": str(e)}


register_tool(
    name="csv_profile",
    description="Profile a CSV file: column names/types, row count, sample rows. Uses DuckDB.",
    parameters={
        "type": "object",
        "properties": {
            "file_id": {"type": "integer", "description": "ID of the CSV File record."},
        },
        "required": ["file_id"],
    },
    handler=_csv_profile,
)


# ---------------------------------------------------------------------------
# csv.query
# ---------------------------------------------------------------------------
def _csv_query(session: Session, run_id: int, *, file_id: int, sql: str) -> dict:
    """Execute read-only SQL on a CSV via DuckDB. Table alias is 'df'."""
    from sred.ingest.csv_intel import csv_query

    file = session.get(File, file_id)
    if not file:
        return {"error": f"File {file_id} not found."}
    if file.run_id != run_id:
        return {"error": "File does not belong to the active run."}

    file_path = str(DATA_DIR / file.path)
    rows = csv_query(file_path, sql)
    # Cap output to avoid token explosion
    if len(rows) > 50:
        rows = rows[:50]
        return {"rows": rows, "truncated": True, "total_returned": 50}
    return {"rows": rows, "truncated": False, "total_returned": len(rows)}


register_tool(
    name="csv_query",
    description="Run read-only SQL on a CSV file via DuckDB. Reference the table as 'df'. Example: SELECT * FROM df WHERE hours > 5 LIMIT 10",
    parameters={
        "type": "object",
        "properties": {
            "file_id": {"type": "integer", "description": "ID of the CSV File record."},
            "sql": {"type": "string", "description": "SQL query. Use 'df' as the table name."},
        },
        "required": ["file_id", "sql"],
    },
    handler=_csv_query,
)


# ---------------------------------------------------------------------------
# people.list
# ---------------------------------------------------------------------------
def _people_list(session: Session, run_id: int) -> dict:
    """List all people in the current run."""
    people = session.exec(select(Person).where(Person.run_id == run_id)).all()
    return {
        "count": len(people),
        "people": [
            {
                "id": p.id,
                "name": p.name,
                "role": p.role,
                "hourly_rate": p.hourly_rate,
                "rate_status": p.rate_status.value,
            }
            for p in people
        ],
    }


register_tool(
    name="people_list",
    description="List all people (employees/contractors) in the current run with their roles and rates.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_people_list,
)


# ---------------------------------------------------------------------------
# people.get
# ---------------------------------------------------------------------------
def _people_get(session: Session, run_id: int, *, person_id: int) -> dict:
    """Get details for a single person."""
    person = session.get(Person, person_id)
    if not person:
        return {"error": f"Person {person_id} not found."}
    if person.run_id != run_id:
        return {"error": "Person does not belong to the active run."}
    return {
        "id": person.id,
        "name": person.name,
        "role": person.role,
        "hourly_rate": person.hourly_rate,
        "rate_status": person.rate_status.value,
        "email": person.email,
    }


register_tool(
    name="people_get",
    description="Get detailed information about a specific person by ID.",
    parameters={
        "type": "object",
        "properties": {
            "person_id": {"type": "integer", "description": "ID of the Person record."},
        },
        "required": ["person_id"],
    },
    handler=_people_get,
)


# ---------------------------------------------------------------------------
# tasks.list_open  (ReviewTask-based)
# ---------------------------------------------------------------------------
def _tasks_list_open(session: Session, run_id: int, *, severity: str = "all") -> dict:
    """List open review tasks for the run, optionally filtered by severity."""
    stmt = select(ReviewTask).where(
        ReviewTask.run_id == run_id,
        ReviewTask.status == ReviewTaskStatus.OPEN,
    )
    if severity != "all":
        sev = ContradictionSeverity(severity) if severity in ContradictionSeverity.__members__ else None
        if sev:
            stmt = stmt.where(ReviewTask.severity == sev)

    tasks = session.exec(stmt).all()
    items = [
        {
            "id": t.id,
            "issue_key": t.issue_key,
            "title": t.title,
            "severity": t.severity.value,
            "status": t.status.value,
        }
        for t in tasks
    ]
    return {"count": len(items), "tasks": items}


register_tool(
    name="tasks_list_open",
    description="List open review tasks for the current run. Optionally filter by severity (LOW, MEDIUM, HIGH, BLOCKING).",
    parameters={
        "type": "object",
        "properties": {
            "severity": {"type": "string", "description": "Filter: LOW, MEDIUM, HIGH, BLOCKING, or 'all' (default)."},
        },
        "required": [],
    },
    handler=_tasks_list_open,
)


# ---------------------------------------------------------------------------
# tasks.create  (ReviewTask-based, deduped by issue_key, lock-aware)
# ---------------------------------------------------------------------------
def _tasks_create(
    session: Session,
    run_id: int,
    *,
    issue_key: str,
    title: str,
    description: str,
    severity: str = "MEDIUM",
    contradiction_id: int | None = None,
) -> dict:
    """Create a review task. Deduped by issue_key; blocked if a lock exists."""
    # Check lock
    if has_active_lock(session, run_id, issue_key):
        return {"status": "locked", "issue_key": issue_key, "message": "A DecisionLock exists for this issue. Cannot re-open."}

    # Check existing open task with same key
    existing = session.exec(
        select(ReviewTask).where(
            ReviewTask.run_id == run_id,
            ReviewTask.issue_key == issue_key,
            ReviewTask.status == ReviewTaskStatus.OPEN,
        )
    ).first()
    if existing:
        return {"status": "duplicate", "task_id": existing.id, "issue_key": issue_key}

    sev = ContradictionSeverity(severity) if severity in ContradictionSeverity.__members__ else ContradictionSeverity.MEDIUM
    task = ReviewTask(
        run_id=run_id,
        issue_key=issue_key,
        title=title,
        description=description,
        severity=sev,
        contradiction_id=contradiction_id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)

    # Re-evaluate gate
    update_run_gate_status(session, run_id)

    return {"status": "created", "task_id": task.id, "issue_key": issue_key}


register_tool(
    name="tasks_create",
    description="Create a review task for human attention. Deduped by issue_key; blocked if a DecisionLock exists for that key.",
    parameters={
        "type": "object",
        "properties": {
            "issue_key": {"type": "string", "description": "Unique key, e.g. 'MISSING_RATE:person:3'."},
            "title": {"type": "string", "description": "Short title."},
            "description": {"type": "string", "description": "Detailed description."},
            "severity": {"type": "string", "description": "LOW, MEDIUM, HIGH, or BLOCKING. Default MEDIUM."},
            "contradiction_id": {"type": "integer", "description": "Optional linked Contradiction ID."},
        },
        "required": ["issue_key", "title", "description"],
    },
    handler=_tasks_create,
)


# ---------------------------------------------------------------------------
# contradictions.list_open  (Contradiction model)
# ---------------------------------------------------------------------------
def _contradictions_list_open(session: Session, run_id: int) -> dict:
    """List open contradictions."""
    items = session.exec(
        select(Contradiction).where(
            Contradiction.run_id == run_id,
            Contradiction.status == ContradictionStatus.OPEN,
        )
    ).all()
    return {
        "count": len(items),
        "contradictions": [
            {
                "id": c.id,
                "issue_key": c.issue_key,
                "type": c.contradiction_type.value,
                "severity": c.severity.value,
                "description": c.description,
            }
            for c in items
        ],
    }


register_tool(
    name="contradictions_list_open",
    description="List open contradictions for the current run.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_contradictions_list_open,
)


# ---------------------------------------------------------------------------
# contradictions.create  (Contradiction model, deduped by issue_key, lock-aware)
# ---------------------------------------------------------------------------
def _contradictions_create(
    session: Session,
    run_id: int,
    *,
    issue_key: str,
    contradiction_type: str,
    severity: str,
    description: str,
    related_entity_type: str | None = None,
    related_entity_id: int | None = None,
) -> dict:
    """Create a contradiction. Deduped by issue_key; blocked if a lock exists."""
    if has_active_lock(session, run_id, issue_key):
        return {"status": "locked", "issue_key": issue_key, "message": "A DecisionLock exists. Cannot re-open."}

    existing = session.exec(
        select(Contradiction).where(
            Contradiction.run_id == run_id,
            Contradiction.issue_key == issue_key,
            Contradiction.status == ContradictionStatus.OPEN,
        )
    ).first()
    if existing:
        return {"status": "duplicate", "contradiction_id": existing.id, "issue_key": issue_key}

    ct = ContradictionType(contradiction_type) if contradiction_type in ContradictionType.__members__ else ContradictionType.OTHER
    sev = ContradictionSeverity(severity) if severity in ContradictionSeverity.__members__ else ContradictionSeverity.MEDIUM

    c = Contradiction(
        run_id=run_id,
        issue_key=issue_key,
        contradiction_type=ct,
        severity=sev,
        description=description,
        related_entity_type=related_entity_type,
        related_entity_id=related_entity_id,
    )
    session.add(c)
    session.commit()
    session.refresh(c)

    update_run_gate_status(session, run_id)

    return {"status": "created", "contradiction_id": c.id, "issue_key": issue_key}


register_tool(
    name="contradictions_create",
    description="Flag a contradiction/data conflict. Deduped by issue_key; blocked if a DecisionLock exists.",
    parameters={
        "type": "object",
        "properties": {
            "issue_key": {"type": "string", "description": "Unique key, e.g. 'PAYROLL_MISMATCH:2024-Q1'."},
            "contradiction_type": {"type": "string", "description": "MISSING_RATE, PAYROLL_MISMATCH, UNKNOWN_BASIS, MISSING_EVIDENCE, or OTHER."},
            "severity": {"type": "string", "description": "LOW, MEDIUM, HIGH, or BLOCKING."},
            "description": {"type": "string", "description": "Detailed description."},
            "related_entity_type": {"type": "string", "description": "Optional: entity type, e.g. 'Person'."},
            "related_entity_id": {"type": "integer", "description": "Optional: entity ID."},
        },
        "required": ["issue_key", "contradiction_type", "severity", "description"],
    },
    handler=_contradictions_create,
)


# ---------------------------------------------------------------------------
# locks.list_active  (DecisionLock model)
# ---------------------------------------------------------------------------
def _locks_list_active(session: Session, run_id: int) -> dict:
    """List active decision locks."""
    locks = session.exec(
        select(DecisionLock).where(
            DecisionLock.run_id == run_id,
            DecisionLock.active == True,  # noqa: E712
        )
    ).all()
    return {
        "count": len(locks),
        "locks": [
            {"id": lk.id, "issue_key": lk.issue_key, "reason": lk.reason}
            for lk in locks
        ],
    }


register_tool(
    name="locks_list_active",
    description="List active decision locks. The agent must not re-open issues with active locks.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_locks_list_active,
)


# ---------------------------------------------------------------------------
# memory.write_summary
# ---------------------------------------------------------------------------
def _memory_write_summary(session: Session, run_id: int, *, content: str, path: str = "memory/summary.md") -> dict:
    """Write or update a memory document (idempotent by path)."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    existing = session.exec(
        select(MemoryDoc).where(MemoryDoc.run_id == run_id, MemoryDoc.path == path)
    ).first()

    if existing:
        if existing.content_hash == content_hash:
            return {"status": "unchanged", "id": existing.id}
        existing.content_md = content
        existing.content_hash = content_hash
        session.add(existing)
        session.commit()
        return {"status": "updated", "id": existing.id}

    doc = MemoryDoc(
        run_id=run_id,
        path=path,
        content_md=content,
        content_hash=content_hash,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return {"status": "created", "id": doc.id}


register_tool(
    name="memory_write_summary",
    description="Write or update a memory/summary document for the run. Idempotent by path.",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Markdown content to store."},
            "path": {"type": "string", "description": "Logical path, e.g. 'memory/summary.md'. Default: 'memory/summary.md'."},
        },
        "required": ["content"],
    },
    handler=_memory_write_summary,
)
