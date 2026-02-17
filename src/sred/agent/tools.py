"""
Tool implementations for the agent runner.

Each tool function signature: handler(session, run_id, **kwargs) -> dict

Rules:
- LLM never writes SQL or DB directly.
- Tools enforce provenance, idempotency, caching.
"""
import json
import hashlib
from difflib import SequenceMatcher
from typing import Dict, Any, List
from datetime import date as date_type
from sqlmodel import Session, select, func
from sred.models.core import File, Segment, Person, Run, RateStatus
from sred.models.finance import StagingRow, StagingRowType, PayrollExtract
from sred.models.artifact import ExtractionArtifact, ArtifactKind
from sred.models.memory import MemoryDoc
from sred.models.alias import PersonAlias, AliasStatus
from sred.config import settings
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


# ---------------------------------------------------------------------------
# Fuzzy-match helper (stdlib difflib, no extra dependency)
# ---------------------------------------------------------------------------
def _fuzzy_ratio(a: str, b: str) -> float:
    """Case-insensitive similarity ratio between two strings (0-1)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ---------------------------------------------------------------------------
# aliases.resolve — Entity Resolution: StagingRow names → Person
# ---------------------------------------------------------------------------
def _aliases_resolve(
    session: Session,
    run_id: int,
    *,
    person_column: str = "person",
    threshold: float = 0.5,
) -> dict:
    """Extract distinct name values from TIMESHEET StagingRows, fuzzy-match
    against Person records, and return proposals.  Does NOT write anything —
    the caller (or the user) must confirm via ``aliases_confirm``."""

    # 1. Gather all Person records for this run
    people = session.exec(select(Person).where(Person.run_id == run_id)).all()
    if not people:
        return {"error": "No Person records in this run. Add people first."}

    # 2. Gather distinct raw names from StagingRow.raw_data
    staging_rows = session.exec(
        select(StagingRow).where(
            StagingRow.run_id == run_id,
            StagingRow.row_type == StagingRowType.TIMESHEET,
        )
    ).all()
    if not staging_rows:
        return {"error": "No TIMESHEET StagingRows found. Ingest a timesheet CSV first."}

    raw_names: set[str] = set()
    for sr in staging_rows:
        try:
            row_dict = json.loads(sr.raw_data)
        except json.JSONDecodeError:
            continue
        val = row_dict.get(person_column)
        if val and isinstance(val, str) and val.strip():
            raw_names.add(val.strip())

    if not raw_names:
        return {
            "error": f"No values found under column '{person_column}' in TIMESHEET rows. "
                     "Check the column name or run csv_profile first.",
        }

    # 3. Check which aliases already exist (skip already-resolved names)
    existing_aliases = session.exec(
        select(PersonAlias).where(PersonAlias.run_id == run_id)
    ).all()
    existing_alias_set = {ea.alias for ea in existing_aliases}

    # 4. Fuzzy-match each raw name against every Person
    proposals: List[Dict[str, Any]] = []
    for raw_name in sorted(raw_names):
        if raw_name in existing_alias_set:
            continue  # already mapped

        best_person = None
        best_score = 0.0
        for p in people:
            score = _fuzzy_ratio(raw_name, p.name)
            if score > best_score:
                best_score = score
                best_person = p

        if best_person and best_score >= threshold:
            proposals.append({
                "alias": raw_name,
                "person_id": best_person.id,
                "person_name": best_person.name,
                "confidence": round(best_score, 3),
                "status": "auto_match",
            })
        else:
            proposals.append({
                "alias": raw_name,
                "person_id": None,
                "person_name": None,
                "confidence": round(best_score, 3) if best_person else 0.0,
                "status": "no_match",
            })

    return {
        "total_distinct_names": len(raw_names),
        "already_mapped": len(existing_alias_set & raw_names),
        "proposals": proposals,
    }


register_tool(
    name="aliases_resolve",
    description=(
        "Extract distinct person names from TIMESHEET StagingRows, fuzzy-match them "
        "against Person records, and return match proposals. Does NOT write to DB — "
        "use aliases_confirm to persist confirmed links."
    ),
    parameters={
        "type": "object",
        "properties": {
            "person_column": {
                "type": "string",
                "description": "Column name in raw_data JSON that holds the person name. Default: 'person'.",
            },
            "threshold": {
                "type": "number",
                "description": "Minimum fuzzy-match score (0-1) to propose a link. Default: 0.5.",
            },
        },
        "required": [],
    },
    handler=_aliases_resolve,
)


# ---------------------------------------------------------------------------
# aliases.confirm — Persist a confirmed alias link
# ---------------------------------------------------------------------------
def _aliases_confirm(
    session: Session,
    run_id: int,
    *,
    alias: str,
    person_id: int,
    source: str | None = None,
) -> dict:
    """Confirm a name→Person mapping and store it as a PersonAlias.
    Idempotent: if the exact (run_id, alias) already exists, updates it."""

    # Validate person belongs to run
    person = session.get(Person, person_id)
    if not person:
        return {"error": f"Person {person_id} not found."}
    if person.run_id != run_id:
        return {"error": "Person does not belong to the active run."}

    # Check for existing alias in this run
    existing = session.exec(
        select(PersonAlias).where(
            PersonAlias.run_id == run_id,
            PersonAlias.alias == alias,
        )
    ).first()

    confidence = _fuzzy_ratio(alias, person.name)

    if existing:
        existing.person_id = person_id
        existing.confidence = confidence
        existing.status = AliasStatus.CONFIRMED
        existing.source = source
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return {"status": "updated", "alias_id": existing.id, "alias": alias, "person_id": person_id}

    pa = PersonAlias(
        run_id=run_id,
        person_id=person_id,
        alias=alias,
        source=source,
        confidence=confidence,
        status=AliasStatus.CONFIRMED,
    )
    session.add(pa)
    session.commit()
    session.refresh(pa)
    return {"status": "created", "alias_id": pa.id, "alias": alias, "person_id": person_id}


register_tool(
    name="aliases_confirm",
    description=(
        "Confirm a raw-name → Person mapping. Stores a PersonAlias record (CONFIRMED). "
        "Idempotent: updates if the alias already exists for this run."
    ),
    parameters={
        "type": "object",
        "properties": {
            "alias": {"type": "string", "description": "The raw name variant, e.g. 'J. Doe'."},
            "person_id": {"type": "integer", "description": "ID of the Person to link to."},
            "source": {"type": "string", "description": "Optional origin hint, e.g. 'timesheet.csv'."},
        },
        "required": ["alias", "person_id"],
    },
    handler=_aliases_confirm,
)


# ---------------------------------------------------------------------------
# aliases.list — List all aliases for the run
# ---------------------------------------------------------------------------
def _aliases_list(session: Session, run_id: int) -> dict:
    """List all PersonAlias records for the run."""
    aliases = session.exec(
        select(PersonAlias).where(PersonAlias.run_id == run_id)
    ).all()
    return {
        "count": len(aliases),
        "aliases": [
            {
                "id": a.id,
                "alias": a.alias,
                "person_id": a.person_id,
                "confidence": a.confidence,
                "status": a.status.value,
                "source": a.source,
            }
            for a in aliases
        ],
    }


register_tool(
    name="aliases_list",
    description="List all person-name aliases (PersonAlias records) for the current run.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_aliases_list,
)


# ---------------------------------------------------------------------------
# Payroll extraction prompt (structured JSON)
# ---------------------------------------------------------------------------
PAYROLL_EXTRACTION_PROMPT = """\
You are extracting payroll data from a document that may contain multiple sections.
Focus ONLY on sections that contain payroll, labour, salary, or wage information
(e.g. tables titled "Payroll Summary", "Labour Costs", "SR&ED Labour", etc.).
Ignore non-payroll sections such as invoices, cloud costs, or project descriptions.

For each pay period you find, extract:
- period_start: start date (YYYY-MM-DD) — look for "Claim period", "Pay period", date ranges
- period_end: end date (YYYY-MM-DD)
- total_hours: total hours worked across ALL employees. If no explicit total row exists,
  SUM the individual employee hours from columns like "Total Hrs", "Hours", "Hrs Worked".
  Do NOT return null if individual employee hours are listed — always compute the sum.
- total_wages: total wages/salary amount. Use the TOTAL row if present, otherwise sum individual amounts.
- currency: currency code (default "CAD")
- employee_count: number of distinct employees listed
- confidence: your confidence in this extraction (0.0 to 1.0)

Return JSON:
{{
    "periods": [
        {{
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "total_hours": 9024.0,
            "total_wages": 718406.40,
            "currency": "CAD",
            "employee_count": 10,
            "confidence": 0.95
        }}
    ]
}}

If there is absolutely NO payroll, labour, or wage data anywhere in the document, return:
{{"periods": [], "error": "No payroll data found"}}

Document text:
{text}
"""


# ---------------------------------------------------------------------------
# payroll.extract — Structured LLM extraction from vision artifacts
# ---------------------------------------------------------------------------
def _payroll_extract(session: Session, run_id: int, *, file_id: int) -> dict:
    """Extract structured payroll data from a file's vision artifacts using LLM.

    Reads VISION_TEXT ExtractionArtifacts for the file, sends combined text
    to the structured model, and stores PayrollExtract records.
    Idempotent: skips periods that already exist for this file.
    """
    from sred.llm.openai_client import get_chat_completion

    file = session.get(File, file_id)
    if not file:
        return {"error": f"File {file_id} not found."}
    if file.run_id != run_id:
        return {"error": "File does not belong to the active run."}

    # Gather vision text artifacts for this file
    artifacts = session.exec(
        select(ExtractionArtifact).where(
            ExtractionArtifact.file_id == file_id,
            ExtractionArtifact.run_id == run_id,
            ExtractionArtifact.kind == ArtifactKind.VISION_TEXT,
        )
    ).all()

    if not artifacts:
        return {"error": f"No vision text artifacts found for file {file_id}. Process the file first."}

    # Combine all artifact text
    combined_text = "\n\n---\n\n".join(a.data for a in artifacts)

    # Call structured LLM
    prompt = PAYROLL_EXTRACTION_PROMPT.format(text=combined_text[:8000])
    try:
        response_text = get_chat_completion(prompt, json_mode=True)
        data = json.loads(response_text)
    except (json.JSONDecodeError, Exception) as e:
        return {"error": f"LLM extraction failed: {e}"}

    periods = data.get("periods", [])
    if not periods:
        return {"status": "no_periods", "message": data.get("error", "No pay periods found.")}

    created = 0
    skipped = 0
    results = []

    for p in periods:
        try:
            p_start = date_type.fromisoformat(p["period_start"])
            p_end = date_type.fromisoformat(p["period_end"])
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed period: {e}")
            continue

        # Idempotent: check if this period already exists
        existing = session.exec(
            select(PayrollExtract).where(
                PayrollExtract.run_id == run_id,
                PayrollExtract.file_id == file_id,
                PayrollExtract.period_start == p_start,
                PayrollExtract.period_end == p_end,
            )
        ).first()

        if existing:
            skipped += 1
            results.append({"period": f"{p_start}/{p_end}", "status": "exists", "id": existing.id})
            continue

        pe = PayrollExtract(
            run_id=run_id,
            file_id=file_id,
            period_start=p_start,
            period_end=p_end,
            total_hours=p.get("total_hours"),
            total_wages=p.get("total_wages"),
            currency=p.get("currency", "CAD"),
            employee_count=p.get("employee_count"),
            confidence=p.get("confidence", 0.0),
            raw_json=json.dumps(p, default=str),
        )
        session.add(pe)
        session.commit()
        session.refresh(pe)
        created += 1
        results.append({"period": f"{p_start}/{p_end}", "status": "created", "id": pe.id})

    return {"created": created, "skipped": skipped, "periods": results}


register_tool(
    name="payroll_extract",
    description=(
        "Extract structured payroll data (pay periods, hours, wages) from a file's "
        "vision artifacts using the structured LLM. Stores PayrollExtract records. Idempotent."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_id": {"type": "integer", "description": "ID of the payroll File record."},
        },
        "required": ["file_id"],
    },
    handler=_payroll_extract,
)


# ---------------------------------------------------------------------------
# payroll.validate — Compare payroll vs timesheet totals
# ---------------------------------------------------------------------------
def _payroll_validate(
    session: Session,
    run_id: int,
    *,
    hours_column: str = "hours",
) -> dict:
    """Compare payroll extract totals vs timesheet staging row totals.

    Groups timesheet hours by overlapping payroll periods.  If the mismatch
    exceeds PAYROLL_MISMATCH_THRESHOLD (default 5%), creates a BLOCKING
    Contradiction and ReviewTask.
    """
    threshold = settings.PAYROLL_MISMATCH_THRESHOLD

    # 1. Load payroll extracts
    payroll_extracts = session.exec(
        select(PayrollExtract).where(PayrollExtract.run_id == run_id)
    ).all()
    if not payroll_extracts:
        return {"error": "No PayrollExtract records found. Run payroll_extract first."}

    # 2. Load timesheet staging rows
    ts_rows = session.exec(
        select(StagingRow).where(
            StagingRow.run_id == run_id,
            StagingRow.row_type == StagingRowType.TIMESHEET,
        )
    ).all()
    if not ts_rows:
        return {"error": "No TIMESHEET StagingRows found. Ingest a timesheet CSV first."}

    # 3. Sum timesheet hours per date
    ts_hours_by_date: Dict[str, float] = {}  # date-string -> total hours
    ts_total_hours = 0.0
    for sr in ts_rows:
        try:
            row_dict = json.loads(sr.raw_data)
        except json.JSONDecodeError:
            continue
        h = row_dict.get(hours_column)
        d = row_dict.get("date")
        if h is not None:
            try:
                hours_val = float(h)
            except (ValueError, TypeError):
                continue
            ts_total_hours += hours_val
            if d:
                ts_hours_by_date[str(d)] = ts_hours_by_date.get(str(d), 0.0) + hours_val

    # 4. Compare per payroll period
    comparisons: List[Dict[str, Any]] = []
    overall_payroll_hours = 0.0
    contradictions_created = 0

    for pe in payroll_extracts:
        if pe.total_hours is None:
            comparisons.append({
                "period": f"{pe.period_start}/{pe.period_end}",
                "payroll_hours": None,
                "timesheet_hours": None,
                "status": "no_payroll_hours",
            })
            continue

        overall_payroll_hours += pe.total_hours

        # Sum timesheet hours that fall within this period
        period_ts_hours = 0.0
        for date_str, hrs in ts_hours_by_date.items():
            try:
                d = date_type.fromisoformat(date_str)
            except ValueError:
                continue
            if pe.period_start <= d <= pe.period_end:
                period_ts_hours += hrs

        # Compute mismatch
        if pe.total_hours == 0 and period_ts_hours == 0:
            mismatch_pct = 0.0
        elif pe.total_hours == 0:
            mismatch_pct = 1.0  # 100% mismatch
        else:
            mismatch_pct = abs(pe.total_hours - period_ts_hours) / pe.total_hours

        is_blocking = mismatch_pct > threshold

        comp = {
            "period": f"{pe.period_start}/{pe.period_end}",
            "payroll_hours": pe.total_hours,
            "timesheet_hours": round(period_ts_hours, 2),
            "mismatch_pct": round(mismatch_pct * 100, 2),
            "threshold_pct": round(threshold * 100, 2),
            "blocking": is_blocking,
        }

        if is_blocking:
            issue_key = f"PAYROLL_MISMATCH:{pe.period_start}/{pe.period_end}"

            # Create contradiction (uses dedup logic)
            if not has_active_lock(session, run_id, issue_key):
                existing_c = session.exec(
                    select(Contradiction).where(
                        Contradiction.run_id == run_id,
                        Contradiction.issue_key == issue_key,
                        Contradiction.status == ContradictionStatus.OPEN,
                    )
                ).first()
                if not existing_c:
                    c = Contradiction(
                        run_id=run_id,
                        issue_key=issue_key,
                        contradiction_type=ContradictionType.PAYROLL_MISMATCH,
                        severity=ContradictionSeverity.BLOCKING,
                        description=(
                            f"Payroll vs timesheet mismatch for {pe.period_start} to {pe.period_end}: "
                            f"payroll={pe.total_hours}h, timesheet={period_ts_hours:.1f}h, "
                            f"diff={mismatch_pct*100:.1f}% (threshold {threshold*100:.0f}%)"
                        ),
                    )
                    session.add(c)
                    session.commit()
                    session.refresh(c)

                    # Create linked ReviewTask
                    task = ReviewTask(
                        run_id=run_id,
                        issue_key=issue_key,
                        title=f"Payroll mismatch: {pe.period_start} to {pe.period_end}",
                        description=c.description,
                        severity=ContradictionSeverity.BLOCKING,
                        contradiction_id=c.id,
                    )
                    session.add(task)
                    session.commit()

                    update_run_gate_status(session, run_id)
                    contradictions_created += 1
                    comp["contradiction"] = "created"
                else:
                    comp["contradiction"] = "duplicate"
            else:
                comp["contradiction"] = "locked"

        comparisons.append(comp)

    # 5. Overall summary
    if overall_payroll_hours == 0 and ts_total_hours == 0:
        overall_mismatch = 0.0
    elif overall_payroll_hours == 0:
        overall_mismatch = 1.0
    else:
        overall_mismatch = abs(overall_payroll_hours - ts_total_hours) / overall_payroll_hours

    return {
        "overall_payroll_hours": round(overall_payroll_hours, 2),
        "overall_timesheet_hours": round(ts_total_hours, 2),
        "overall_mismatch_pct": round(overall_mismatch * 100, 2),
        "threshold_pct": round(threshold * 100, 2),
        "overall_blocking": overall_mismatch > threshold,
        "contradictions_created": contradictions_created,
        "period_comparisons": comparisons,
    }


register_tool(
    name="payroll_validate",
    description=(
        "Compare payroll extract totals vs timesheet staging row totals per period. "
        "If mismatch exceeds 5% threshold, creates a BLOCKING contradiction and ReviewTask."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hours_column": {
                "type": "string",
                "description": "Column name in timesheet raw_data JSON for hours. Default: 'hours'.",
            },
        },
        "required": [],
    },
    handler=_payroll_validate,
)


# ---------------------------------------------------------------------------
# payroll.summary — List stored payroll extracts
# ---------------------------------------------------------------------------
def _payroll_summary(session: Session, run_id: int) -> dict:
    """List all PayrollExtract records for the run."""
    extracts = session.exec(
        select(PayrollExtract).where(PayrollExtract.run_id == run_id)
    ).all()
    return {
        "count": len(extracts),
        "extracts": [
            {
                "id": e.id,
                "file_id": e.file_id,
                "period_start": str(e.period_start),
                "period_end": str(e.period_end),
                "total_hours": e.total_hours,
                "total_wages": e.total_wages,
                "currency": e.currency,
                "employee_count": e.employee_count,
                "confidence": e.confidence,
            }
            for e in extracts
        ],
    }


register_tool(
    name="payroll_summary",
    description="List all extracted payroll period records for the current run.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_payroll_summary,
)
