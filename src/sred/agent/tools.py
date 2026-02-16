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
# tasks.list_open
# ---------------------------------------------------------------------------
def _tasks_list_open(session: Session, run_id: int, *, severity: str = "all") -> dict:
    """List open hypotheses (tasks) for the run, optionally filtered by status."""
    stmt = select(Hypothesis).where(
        Hypothesis.run_id == run_id,
        Hypothesis.status == HypothesisStatus.ACTIVE,
    )
    hyps = session.exec(stmt).all()
    items = []
    for h in hyps:
        items.append({
            "id": h.id,
            "type": h.type.value,
            "description": h.description,
            "status": h.status.value,
        })
    return {"count": len(items), "tasks": items}


register_tool(
    name="tasks_list_open",
    description="List open/active tasks (hypotheses) for the current run.",
    parameters={
        "type": "object",
        "properties": {
            "severity": {"type": "string", "description": "Filter by severity (unused for now, returns all active)."},
        },
        "required": [],
    },
    handler=_tasks_list_open,
)


# ---------------------------------------------------------------------------
# tasks.create
# ---------------------------------------------------------------------------
def _tasks_create(session: Session, run_id: int, *, task_type: str, description: str) -> dict:
    """Create a new hypothesis/task."""
    from sred.models.hypothesis import HypothesisType

    # Validate type
    valid_types = {t.value for t in HypothesisType}
    if task_type not in valid_types:
        return {"error": f"Invalid task_type '{task_type}'. Valid: {sorted(valid_types)}"}

    hyp = Hypothesis(
        run_id=run_id,
        type=HypothesisType(task_type),
        description=description,
        status=HypothesisStatus.ACTIVE,
    )
    session.add(hyp)
    session.commit()
    session.refresh(hyp)
    return {"id": hyp.id, "type": hyp.type.value, "description": hyp.description}


register_tool(
    name="tasks_create",
    description="Create a new task (hypothesis) for the current run.",
    parameters={
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "description": "Type of task: CSV_SCHEMA or CLAIM_CLUSTERING."},
            "description": {"type": "string", "description": "Human-readable description of the task."},
        },
        "required": ["task_type", "description"],
    },
    handler=_tasks_create,
)


# ---------------------------------------------------------------------------
# contradictions.list_open
# ---------------------------------------------------------------------------
def _contradictions_list_open(session: Session, run_id: int) -> dict:
    """List open contradictions (REJECTED hypotheses that need attention)."""
    stmt = select(Hypothesis).where(
        Hypothesis.run_id == run_id,
        Hypothesis.status == HypothesisStatus.REJECTED,
    )
    hyps = session.exec(stmt).all()
    items = [
        {"id": h.id, "type": h.type.value, "description": h.description}
        for h in hyps
    ]
    return {"count": len(items), "contradictions": items}


register_tool(
    name="contradictions_list_open",
    description="List open contradictions (rejected hypotheses) that may need resolution.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_contradictions_list_open,
)


# ---------------------------------------------------------------------------
# contradictions.create
# ---------------------------------------------------------------------------
def _contradictions_create(session: Session, run_id: int, *, description: str, parent_id: int | None = None) -> dict:
    """Create a contradiction record (a rejected hypothesis)."""
    from sred.models.hypothesis import HypothesisType

    hyp = Hypothesis(
        run_id=run_id,
        type=HypothesisType.CLAIM_CLUSTERING,
        description=f"[CONTRADICTION] {description}",
        status=HypothesisStatus.REJECTED,
        parent_id=parent_id,
    )
    session.add(hyp)
    session.commit()
    session.refresh(hyp)
    return {"id": hyp.id, "description": hyp.description}


register_tool(
    name="contradictions_create",
    description="Flag a contradiction or data conflict for human review.",
    parameters={
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Description of the contradiction."},
            "parent_id": {"type": "integer", "description": "Optional parent hypothesis ID."},
        },
        "required": ["description"],
    },
    handler=_contradictions_create,
)


# ---------------------------------------------------------------------------
# locks.list_active
# ---------------------------------------------------------------------------
def _locks_list_active(session: Session, run_id: int) -> dict:
    """List active locks (accepted hypotheses that are frozen)."""
    stmt = select(Hypothesis).where(
        Hypothesis.run_id == run_id,
        Hypothesis.status == HypothesisStatus.ACCEPTED,
    )
    hyps = session.exec(stmt).all()
    items = [
        {"id": h.id, "type": h.type.value, "description": h.description}
        for h in hyps
    ]
    return {"count": len(items), "locks": items}


register_tool(
    name="locks_list_active",
    description="List accepted/locked hypotheses that should not be modified.",
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
