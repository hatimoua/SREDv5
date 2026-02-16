import pytest
import json
import hashlib
from unittest.mock import MagicMock, patch
from sqlmodel import Session, SQLModel, create_engine, select
from sred.models.core import Run, RunStatus, File, Person, RateStatus, Segment, SegmentStatus
from sred.models.hypothesis import Hypothesis, HypothesisType, HypothesisStatus
from sred.models.memory import MemoryDoc
from sred.models.agent_log import ToolCallLog, LLMCallLog
from sred.models.world import (
    Contradiction, ContradictionSeverity, ContradictionType, ContradictionStatus,
    ReviewTask, ReviewTaskStatus, ReviewDecision, DecisionLock,
)
from sred.gates import (
    get_blocking_contradictions, get_open_blocking_tasks,
    has_active_lock, update_run_gate_status,
)
from sred.agent.registry import get_openai_tools_schema, get_tool_handler, TOOL_REGISTRY
from sred.models.alias import PersonAlias, AliasStatus
from sred.models.finance import StagingRow, StagingRowType, PayrollExtract
from sred.models.artifact import ExtractionArtifact, ArtifactKind
from sred.models.core import FileStatus
from datetime import date
from sred.agent.tools import (
    _people_list,
    _people_get,
    _tasks_list_open,
    _tasks_create,
    _contradictions_list_open,
    _contradictions_create,
    _locks_list_active,
    _memory_write_summary,
    _aliases_resolve,
    _aliases_confirm,
    _aliases_list,
    _fuzzy_ratio,
    _payroll_extract,
    _payroll_validate,
    _payroll_summary,
)


@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def run(session):
    r = Run(name="Agent Test Run")
    session.add(r)
    session.commit()
    session.refresh(r)
    return r


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------
def test_registry_has_all_tools():
    expected = {
        "ingest_process_file",
        "search_hybrid",
        "csv_profile",
        "csv_query",
        "people_list",
        "people_get",
        "tasks_list_open",
        "tasks_create",
        "contradictions_list_open",
        "contradictions_create",
        "locks_list_active",
        "memory_write_summary",
        "aliases_resolve",
        "aliases_confirm",
        "aliases_list",
        "payroll_extract",
        "payroll_validate",
        "payroll_summary",
    }
    assert expected.issubset(set(TOOL_REGISTRY.keys()))


def test_openai_schema_format():
    schema = get_openai_tools_schema()
    assert len(schema) >= 18
    for tool in schema:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


# ---------------------------------------------------------------------------
# people tools
# ---------------------------------------------------------------------------
def test_people_list_empty(session, run):
    result = _people_list(session, run.id)
    assert result["count"] == 0
    assert result["people"] == []


def test_people_list_with_data(session, run):
    p = Person(run_id=run.id, name="Alice", role="Dev", hourly_rate=100.0, rate_status=RateStatus.SET)
    session.add(p)
    session.commit()

    result = _people_list(session, run.id)
    assert result["count"] == 1
    assert result["people"][0]["name"] == "Alice"


def test_people_get(session, run):
    p = Person(run_id=run.id, name="Bob", role="QA", hourly_rate=80.0, rate_status=RateStatus.SET)
    session.add(p)
    session.commit()
    session.refresh(p)

    result = _people_get(session, run.id, person_id=p.id)
    assert result["name"] == "Bob"
    assert result["role"] == "QA"


def test_people_get_not_found(session, run):
    result = _people_get(session, run.id, person_id=9999)
    assert "error" in result


def test_people_get_wrong_run(session, run):
    other_run = Run(name="Other")
    session.add(other_run)
    session.commit()
    p = Person(run_id=other_run.id, name="Eve", role="PM")
    session.add(p)
    session.commit()
    session.refresh(p)

    result = _people_get(session, run.id, person_id=p.id)
    assert "error" in result


# ---------------------------------------------------------------------------
# tasks tools (ReviewTask-based)
# ---------------------------------------------------------------------------
def test_tasks_create_and_list(session, run):
    result = _tasks_create(
        session, run.id,
        issue_key="TEST:task:1",
        title="Check CSV mapping",
        description="Verify column mapping is correct",
    )
    assert result["status"] == "created"
    assert "task_id" in result

    listing = _tasks_list_open(session, run.id)
    assert listing["count"] == 1
    assert listing["tasks"][0]["issue_key"] == "TEST:task:1"


def test_tasks_create_dedupe(session, run):
    """Same issue_key should not create a duplicate open task."""
    r1 = _tasks_create(session, run.id, issue_key="DUP:1", title="T", description="D")
    assert r1["status"] == "created"

    r2 = _tasks_create(session, run.id, issue_key="DUP:1", title="T2", description="D2")
    assert r2["status"] == "duplicate"
    assert r2["task_id"] == r1["task_id"]


def test_tasks_create_blocked_by_lock(session, run):
    """Cannot create a task if a DecisionLock exists for the issue_key."""
    # Create a decision + lock manually
    decision = ReviewDecision(run_id=run.id, task_id=0, decision="resolved", decided_by="HUMAN")
    session.add(decision)
    session.commit()
    session.refresh(decision)

    lock = DecisionLock(run_id=run.id, issue_key="LOCKED:1", decision_id=decision.id, reason="Done", active=True)
    session.add(lock)
    session.commit()

    result = _tasks_create(session, run.id, issue_key="LOCKED:1", title="T", description="D")
    assert result["status"] == "locked"


# ---------------------------------------------------------------------------
# contradictions tools (Contradiction model)
# ---------------------------------------------------------------------------
def test_contradictions_create_and_list(session, run):
    result = _contradictions_create(
        session, run.id,
        issue_key="PAYROLL_MISMATCH:Q1",
        contradiction_type="PAYROLL_MISMATCH",
        severity="BLOCKING",
        description="Payroll totals differ by 12%",
    )
    assert result["status"] == "created"
    assert "contradiction_id" in result

    listing = _contradictions_list_open(session, run.id)
    assert listing["count"] == 1
    assert listing["contradictions"][0]["issue_key"] == "PAYROLL_MISMATCH:Q1"


def test_contradictions_create_dedupe(session, run):
    r1 = _contradictions_create(
        session, run.id, issue_key="X:1",
        contradiction_type="OTHER", severity="LOW", description="A",
    )
    r2 = _contradictions_create(
        session, run.id, issue_key="X:1",
        contradiction_type="OTHER", severity="LOW", description="B",
    )
    assert r1["status"] == "created"
    assert r2["status"] == "duplicate"


def test_contradictions_create_blocked_by_lock(session, run):
    decision = ReviewDecision(run_id=run.id, task_id=0, decision="ok", decided_by="HUMAN")
    session.add(decision)
    session.commit()
    session.refresh(decision)

    lock = DecisionLock(run_id=run.id, issue_key="LOCKED:C", decision_id=decision.id, reason="Done", active=True)
    session.add(lock)
    session.commit()

    result = _contradictions_create(
        session, run.id, issue_key="LOCKED:C",
        contradiction_type="OTHER", severity="LOW", description="X",
    )
    assert result["status"] == "locked"


# ---------------------------------------------------------------------------
# locks tool (DecisionLock model)
# ---------------------------------------------------------------------------
def test_locks_list(session, run):
    decision = ReviewDecision(run_id=run.id, task_id=0, decision="ok", decided_by="HUMAN")
    session.add(decision)
    session.commit()
    session.refresh(decision)

    lock = DecisionLock(run_id=run.id, issue_key="LOCK:1", decision_id=decision.id, reason="Approved", active=True)
    session.add(lock)
    session.commit()

    result = _locks_list_active(session, run.id)
    assert result["count"] == 1
    assert result["locks"][0]["issue_key"] == "LOCK:1"


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------
def test_gate_no_blockers(session, run):
    """No blockers -> status stays as-is."""
    status = update_run_gate_status(session, run.id)
    assert status == RunStatus.INITIALIZING  # unchanged


def test_gate_blocking_contradiction_triggers_needs_review(session, run):
    c = Contradiction(
        run_id=run.id, issue_key="BLOCK:1",
        contradiction_type=ContradictionType.PAYROLL_MISMATCH,
        severity=ContradictionSeverity.BLOCKING,
        description="Big mismatch",
    )
    session.add(c)
    session.commit()

    status = update_run_gate_status(session, run.id)
    assert status == RunStatus.NEEDS_REVIEW

    session.refresh(run)
    assert run.status == RunStatus.NEEDS_REVIEW


def test_gate_blocking_task_triggers_needs_review(session, run):
    task = ReviewTask(
        run_id=run.id, issue_key="BLOCK:T1",
        title="Fix rate", description="Missing rate",
        severity=ContradictionSeverity.BLOCKING,
    )
    session.add(task)
    session.commit()

    status = update_run_gate_status(session, run.id)
    assert status == RunStatus.NEEDS_REVIEW


def test_gate_resolving_blockers_clears_needs_review(session, run):
    c = Contradiction(
        run_id=run.id, issue_key="BLOCK:2",
        contradiction_type=ContradictionType.MISSING_RATE,
        severity=ContradictionSeverity.BLOCKING,
        description="No rate for Alice",
    )
    session.add(c)
    session.commit()

    update_run_gate_status(session, run.id)
    session.refresh(run)
    assert run.status == RunStatus.NEEDS_REVIEW

    # Resolve it
    c.status = ContradictionStatus.RESOLVED
    session.add(c)
    session.commit()

    status = update_run_gate_status(session, run.id)
    assert status == RunStatus.PROCESSING


def test_has_active_lock(session, run):
    assert not has_active_lock(session, run.id, "KEY:1")

    decision = ReviewDecision(run_id=run.id, task_id=0, decision="ok", decided_by="HUMAN")
    session.add(decision)
    session.commit()
    session.refresh(decision)

    lock = DecisionLock(run_id=run.id, issue_key="KEY:1", decision_id=decision.id, reason="Done", active=True)
    session.add(lock)
    session.commit()

    assert has_active_lock(session, run.id, "KEY:1")
    assert not has_active_lock(session, run.id, "KEY:2")


def test_supersede_lock(session, run):
    """Superseding a lock deactivates old, creates new active lock."""
    decision1 = ReviewDecision(run_id=run.id, task_id=0, decision="first", decided_by="HUMAN")
    session.add(decision1)
    session.commit()
    session.refresh(decision1)

    lock1 = DecisionLock(run_id=run.id, issue_key="SUP:1", decision_id=decision1.id, reason="First", active=True)
    session.add(lock1)
    session.commit()
    session.refresh(lock1)

    # Supersede
    lock1.active = False
    session.add(lock1)

    decision2 = ReviewDecision(run_id=run.id, task_id=0, decision="override", decided_by="HUMAN")
    session.add(decision2)
    session.commit()
    session.refresh(decision2)

    lock2 = DecisionLock(run_id=run.id, issue_key="SUP:1", decision_id=decision2.id, reason="Override", active=True)
    session.add(lock2)
    session.commit()

    # Old lock inactive, new lock active
    session.refresh(lock1)
    assert lock1.active is False
    assert lock2.active is True
    assert has_active_lock(session, run.id, "SUP:1")


def test_non_blocking_contradiction_does_not_trigger_gate(session, run):
    """LOW severity contradiction should not trigger NEEDS_REVIEW."""
    c = Contradiction(
        run_id=run.id, issue_key="LOW:1",
        contradiction_type=ContradictionType.OTHER,
        severity=ContradictionSeverity.LOW,
        description="Minor issue",
    )
    session.add(c)
    session.commit()

    status = update_run_gate_status(session, run.id)
    assert status != RunStatus.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# memory tool
# ---------------------------------------------------------------------------
def test_memory_write_create(session, run):
    result = _memory_write_summary(session, run.id, content="# Summary\nAll good.")
    assert result["status"] == "created"

    doc = session.exec(select(MemoryDoc).where(MemoryDoc.run_id == run.id)).first()
    assert doc is not None
    assert doc.content_md == "# Summary\nAll good."


def test_memory_write_idempotent(session, run):
    r1 = _memory_write_summary(session, run.id, content="Version 1")
    assert r1["status"] == "created"

    r2 = _memory_write_summary(session, run.id, content="Version 1")
    assert r2["status"] == "unchanged"
    assert r2["id"] == r1["id"]


def test_memory_write_update(session, run):
    r1 = _memory_write_summary(session, run.id, content="Version 1")
    r2 = _memory_write_summary(session, run.id, content="Version 2")
    assert r2["status"] == "updated"
    assert r2["id"] == r1["id"]

    doc = session.get(MemoryDoc, r1["id"])
    assert doc.content_md == "Version 2"


# ---------------------------------------------------------------------------
# Agent runner (mocked OpenAI)
# ---------------------------------------------------------------------------
def test_agent_loop_simple_answer(session, run):
    """Agent returns a plain text answer with no tool calls."""
    from sred.agent.runner import run_agent_loop

    mock_message = MagicMock()
    mock_message.content = "Here is my analysis."
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    with patch("sred.agent.runner.client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = run_agent_loop(session, run.id, "Hello", max_steps=3)

    assert result.stopped_reason == "complete"
    assert result.final_answer == "Here is my analysis."
    assert result.total_steps == 1

    # Check LLM log was written
    llm_log = session.exec(select(LLMCallLog).where(LLMCallLog.run_id == run.id)).first()
    assert llm_log is not None
    assert llm_log.total_tokens == 15


def test_agent_loop_with_tool_call(session, run):
    """Agent calls people_list tool, then gives final answer."""
    from sred.agent.runner import run_agent_loop

    # Add a person so the tool returns data
    p = Person(run_id=run.id, name="Alice", role="Dev", hourly_rate=100.0, rate_status=RateStatus.SET)
    session.add(p)
    session.commit()

    # --- First response: tool call ---
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "people_list"
    mock_tc.function.arguments = "{}"

    mock_msg1 = MagicMock()
    mock_msg1.content = "Let me check the people."
    mock_msg1.tool_calls = [mock_tc]

    mock_choice1 = MagicMock()
    mock_choice1.message = mock_msg1
    mock_choice1.finish_reason = "tool_calls"

    mock_usage1 = MagicMock()
    mock_usage1.prompt_tokens = 20
    mock_usage1.completion_tokens = 10
    mock_usage1.total_tokens = 30

    mock_response1 = MagicMock()
    mock_response1.choices = [mock_choice1]
    mock_response1.usage = mock_usage1

    # --- Second response: final answer ---
    mock_msg2 = MagicMock()
    mock_msg2.content = "Found 1 person: Alice (Dev)."
    mock_msg2.tool_calls = None

    mock_choice2 = MagicMock()
    mock_choice2.message = mock_msg2
    mock_choice2.finish_reason = "stop"

    mock_usage2 = MagicMock()
    mock_usage2.prompt_tokens = 40
    mock_usage2.completion_tokens = 15
    mock_usage2.total_tokens = 55

    mock_response2 = MagicMock()
    mock_response2.choices = [mock_choice2]
    mock_response2.usage = mock_usage2

    with patch("sred.agent.runner.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        result = run_agent_loop(session, run.id, "List all people", max_steps=5)

    assert result.stopped_reason == "complete"
    assert "Alice" in result.final_answer
    assert result.total_steps == 2

    # Check tool call log
    tool_log = session.exec(select(ToolCallLog).where(ToolCallLog.run_id == run.id)).first()
    assert tool_log is not None
    assert tool_log.tool_name == "people_list"
    assert tool_log.success is True

    # Check LLM logs (should be 2)
    llm_logs = session.exec(select(LLMCallLog).where(LLMCallLog.run_id == run.id)).all()
    assert len(llm_logs) == 2


def test_agent_loop_max_steps(session, run):
    """Agent hits max_steps limit."""
    from sred.agent.runner import run_agent_loop

    # Always return a tool call so we never finish
    mock_tc = MagicMock()
    mock_tc.id = "call_loop"
    mock_tc.function.name = "people_list"
    mock_tc.function.arguments = "{}"

    mock_msg = MagicMock()
    mock_msg.content = "Checking again..."
    mock_msg.tool_calls = [mock_tc]

    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "tool_calls"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    with patch("sred.agent.runner.client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = run_agent_loop(session, run.id, "Loop forever", max_steps=2)

    assert result.stopped_reason == "max_steps"


def test_agent_loop_unknown_tool(session, run):
    """Agent calls a tool that doesn't exist — error is captured gracefully."""
    from sred.agent.runner import run_agent_loop

    # First response: call unknown tool
    mock_tc = MagicMock()
    mock_tc.id = "call_bad"
    mock_tc.function.name = "nonexistent_tool"
    mock_tc.function.arguments = "{}"

    mock_msg1 = MagicMock()
    mock_msg1.content = None
    mock_msg1.tool_calls = [mock_tc]

    mock_choice1 = MagicMock()
    mock_choice1.message = mock_msg1
    mock_choice1.finish_reason = "tool_calls"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response1 = MagicMock()
    mock_response1.choices = [mock_choice1]
    mock_response1.usage = mock_usage

    # Second response: final answer after error
    mock_msg2 = MagicMock()
    mock_msg2.content = "That tool failed."
    mock_msg2.tool_calls = None

    mock_choice2 = MagicMock()
    mock_choice2.message = mock_msg2
    mock_choice2.finish_reason = "stop"

    mock_response2 = MagicMock()
    mock_response2.choices = [mock_choice2]
    mock_response2.usage = mock_usage

    with patch("sred.agent.runner.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        result = run_agent_loop(session, run.id, "Use bad tool", max_steps=5)

    assert result.stopped_reason == "complete"

    # The failed tool call should be logged with success=False
    tool_log = session.exec(select(ToolCallLog).where(ToolCallLog.run_id == run.id)).first()
    assert tool_log is not None
    assert tool_log.success is False
    assert "Unknown tool" in tool_log.result_json


# ---------------------------------------------------------------------------
# Fuzzy ratio helper
# ---------------------------------------------------------------------------
def test_fuzzy_ratio_exact():
    assert _fuzzy_ratio("John Doe", "John Doe") == 1.0


def test_fuzzy_ratio_case_insensitive():
    assert _fuzzy_ratio("john doe", "John Doe") == 1.0


def test_fuzzy_ratio_partial():
    score = _fuzzy_ratio("J. Doe", "John Doe")
    assert 0.4 < score < 0.9  # partial match, not perfect


def test_fuzzy_ratio_no_match():
    score = _fuzzy_ratio("Alice Smith", "Bob Jones")
    assert score < 0.4


# ---------------------------------------------------------------------------
# aliases tools
# ---------------------------------------------------------------------------
def _make_staging_row(session, run_id, name, source_file_id=None):
    """Helper: create a TIMESHEET StagingRow with a person column."""
    import hashlib as _hl
    raw = json.dumps({"person": name, "date": "2025-01-15", "hours": 8})
    sr = StagingRow(
        run_id=run_id,
        raw_data=raw,
        row_type=StagingRowType.TIMESHEET,
        row_hash=_hl.sha256(raw.encode()).hexdigest(),
        normalized_text=f"{name} 2025-01-15 8h",
        source_file_id=source_file_id,
    )
    session.add(sr)
    session.commit()
    return sr


def test_aliases_resolve_no_people(session, run):
    result = _aliases_resolve(session, run.id)
    assert "error" in result
    assert "No Person" in result["error"]


def test_aliases_resolve_no_staging(session, run):
    session.add(Person(run_id=run.id, name="John Doe", role="Dev"))
    session.commit()
    result = _aliases_resolve(session, run.id)
    assert "error" in result
    assert "No TIMESHEET" in result["error"]


def test_aliases_resolve_exact_match(session, run):
    p = Person(run_id=run.id, name="John Doe", role="Dev")
    session.add(p)
    session.commit()
    session.refresh(p)

    _make_staging_row(session, run.id, "John Doe")

    result = _aliases_resolve(session, run.id)
    assert result["total_distinct_names"] == 1
    assert len(result["proposals"]) == 1
    prop = result["proposals"][0]
    assert prop["alias"] == "John Doe"
    assert prop["person_id"] == p.id
    assert prop["confidence"] == 1.0
    assert prop["status"] == "auto_match"


def test_aliases_resolve_fuzzy_match(session, run):
    p = Person(run_id=run.id, name="John Doe", role="Dev")
    session.add(p)
    session.commit()
    session.refresh(p)

    _make_staging_row(session, run.id, "J. Doe")

    result = _aliases_resolve(session, run.id)
    assert len(result["proposals"]) == 1
    prop = result["proposals"][0]
    assert prop["alias"] == "J. Doe"
    assert prop["person_id"] == p.id
    assert prop["confidence"] > 0.5
    assert prop["status"] == "auto_match"


def test_aliases_resolve_no_match(session, run):
    session.add(Person(run_id=run.id, name="John Doe", role="Dev"))
    session.commit()

    _make_staging_row(session, run.id, "Xyz Abc")

    result = _aliases_resolve(session, run.id, threshold=0.8)
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["status"] == "no_match"


def test_aliases_resolve_skips_existing(session, run):
    p = Person(run_id=run.id, name="John Doe", role="Dev")
    session.add(p)
    session.commit()
    session.refresh(p)

    _make_staging_row(session, run.id, "J. Doe")

    # Pre-create an alias for "J. Doe"
    pa = PersonAlias(run_id=run.id, person_id=p.id, alias="J. Doe", status=AliasStatus.CONFIRMED)
    session.add(pa)
    session.commit()

    result = _aliases_resolve(session, run.id)
    assert result["already_mapped"] == 1
    assert len(result["proposals"]) == 0


def test_aliases_resolve_custom_column(session, run):
    """Resolve should use the specified person_column."""
    p = Person(run_id=run.id, name="Alice", role="Dev")
    session.add(p)
    session.commit()

    # Create a staging row with a non-default column name
    raw = json.dumps({"employee": "Alice", "date": "2025-01-15", "hours": 8})
    sr = StagingRow(
        run_id=run.id,
        raw_data=raw,
        row_type=StagingRowType.TIMESHEET,
        row_hash=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_text="Alice 2025-01-15 8h",
    )
    session.add(sr)
    session.commit()

    # Default column "person" should find nothing
    result_default = _aliases_resolve(session, run.id, person_column="person")
    assert "error" in result_default

    # Custom column "employee" should work
    result = _aliases_resolve(session, run.id, person_column="employee")
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["alias"] == "Alice"


def test_aliases_confirm_create(session, run):
    p = Person(run_id=run.id, name="John Doe", role="Dev")
    session.add(p)
    session.commit()
    session.refresh(p)

    result = _aliases_confirm(session, run.id, alias="J. Doe", person_id=p.id, source="timesheet.csv")
    assert result["status"] == "created"
    assert result["alias"] == "J. Doe"
    assert result["person_id"] == p.id

    # Verify DB record
    pa = session.get(PersonAlias, result["alias_id"])
    assert pa is not None
    assert pa.status == AliasStatus.CONFIRMED
    assert pa.source == "timesheet.csv"
    assert pa.confidence > 0


def test_aliases_confirm_idempotent_update(session, run):
    p1 = Person(run_id=run.id, name="John Doe", role="Dev")
    p2 = Person(run_id=run.id, name="Jane Doe", role="QA")
    session.add_all([p1, p2])
    session.commit()
    session.refresh(p1)
    session.refresh(p2)

    r1 = _aliases_confirm(session, run.id, alias="J. Doe", person_id=p1.id)
    assert r1["status"] == "created"

    # Re-confirm with different person → should update
    r2 = _aliases_confirm(session, run.id, alias="J. Doe", person_id=p2.id)
    assert r2["status"] == "updated"
    assert r2["alias_id"] == r1["alias_id"]
    assert r2["person_id"] == p2.id


def test_aliases_confirm_wrong_run(session, run):
    other_run = Run(name="Other")
    session.add(other_run)
    session.commit()
    p = Person(run_id=other_run.id, name="Eve", role="PM")
    session.add(p)
    session.commit()
    session.refresh(p)

    result = _aliases_confirm(session, run.id, alias="Eve", person_id=p.id)
    assert "error" in result


def test_aliases_confirm_person_not_found(session, run):
    result = _aliases_confirm(session, run.id, alias="Ghost", person_id=9999)
    assert "error" in result


def test_aliases_list_empty(session, run):
    result = _aliases_list(session, run.id)
    assert result["count"] == 0
    assert result["aliases"] == []


def test_aliases_list_with_data(session, run):
    p = Person(run_id=run.id, name="John Doe", role="Dev")
    session.add(p)
    session.commit()
    session.refresh(p)

    _aliases_confirm(session, run.id, alias="J. Doe", person_id=p.id, source="ts.csv")
    _aliases_confirm(session, run.id, alias="Johnny", person_id=p.id)

    result = _aliases_list(session, run.id)
    assert result["count"] == 2
    aliases_set = {a["alias"] for a in result["aliases"]}
    assert aliases_set == {"J. Doe", "Johnny"}


# ---------------------------------------------------------------------------
# build_run_context + dynamic system prompt
# ---------------------------------------------------------------------------
def test_build_run_context_empty_run(session, run):
    from sred.agent.runner import build_run_context

    ctx = build_run_context(session, run.id)
    assert "Agent Test Run" in ctx
    assert "People: 0 total" in ctx
    assert "Files: 0 uploaded" in ctx
    assert "Timesheet staging rows: 0" in ctx
    assert "Person aliases: 0 confirmed / 0 total" in ctx
    assert "Open contradictions: 0" in ctx
    assert "Active decision locks: 0" in ctx


def test_build_run_context_with_data(session, run):
    from sred.agent.runner import build_run_context

    # Add a person with pending rate
    session.add(Person(run_id=run.id, name="Alice", role="Dev"))
    session.commit()

    # Add a staging row
    _make_staging_row(session, run.id, "Alice")

    # Add an open contradiction
    session.add(Contradiction(
        run_id=run.id, issue_key="CTX:1",
        contradiction_type=ContradictionType.OTHER,
        severity=ContradictionSeverity.LOW,
        description="test",
    ))
    session.commit()

    ctx = build_run_context(session, run.id)
    assert "People: 1 total, 1 with PENDING rate" in ctx
    assert "Timesheet staging rows: 1" in ctx
    assert "Open contradictions: 1" in ctx


def test_build_run_context_invalid_run(session):
    from sred.agent.runner import build_run_context

    ctx = build_run_context(session, 9999)
    assert ctx == ""


def test_agent_loop_injects_context_notes(session, run):
    """Verify context_notes appear in the system prompt sent to OpenAI."""
    from sred.agent.runner import run_agent_loop

    mock_message = MagicMock()
    mock_message.content = "Done."
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    with patch("sred.agent.runner.client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        run_agent_loop(
            session, run.id, "Hello",
            max_steps=1,
            context_notes="We are resolving identities for File #12",
        )

        # Inspect the system message that was sent
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]

        # Should contain the static prompt
        assert "SR&ED Automation Agent" in system_msg
        # Should contain the auto-generated run context
        assert "Current Run State" in system_msg
        assert "Agent Test Run" in system_msg
        # Should contain the caller-supplied context notes
        assert "Immediate Goal" in system_msg
        assert "We are resolving identities for File #12" in system_msg


def test_agent_loop_no_context_notes(session, run):
    """Without context_notes, the Immediate Goal section should be absent."""
    from sred.agent.runner import run_agent_loop

    mock_message = MagicMock()
    mock_message.content = "Done."
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    with patch("sred.agent.runner.client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        run_agent_loop(session, run.id, "Hello", max_steps=1)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]

        assert "Current Run State" in system_msg
        assert "Immediate Goal" not in system_msg


# ---------------------------------------------------------------------------
# Payroll tool helpers
# ---------------------------------------------------------------------------
def _make_file(session, run_id, filename="payroll.pdf"):
    """Helper: create a processed File record."""
    f = File(
        run_id=run_id,
        path=f"uploads/{filename}",
        original_filename=filename,
        file_type="application/pdf",
        mime_type="application/pdf",
        size_bytes=1000,
        content_hash=hashlib.sha256(filename.encode()).hexdigest(),
        status=FileStatus.PROCESSED,
    )
    session.add(f)
    session.commit()
    session.refresh(f)
    return f


def _make_artifact(session, file, text="Payroll text"):
    """Helper: create a VISION_TEXT ExtractionArtifact."""
    art = ExtractionArtifact(
        file_id=file.id,
        run_id=file.run_id,
        kind=ArtifactKind.VISION_TEXT,
        data=text,
        model="gpt-4o",
        confidence=1.0,
    )
    session.add(art)
    session.commit()
    return art


# ---------------------------------------------------------------------------
# payroll_extract tests
# ---------------------------------------------------------------------------
def test_payroll_extract_file_not_found(session, run):
    result = _payroll_extract(session, run.id, file_id=9999)
    assert "error" in result


def test_payroll_extract_wrong_run(session, run):
    other_run = Run(name="Other")
    session.add(other_run)
    session.commit()
    f = _make_file(session, other_run.id)
    result = _payroll_extract(session, run.id, file_id=f.id)
    assert "error" in result


def test_payroll_extract_no_artifacts(session, run):
    f = _make_file(session, run.id)
    result = _payroll_extract(session, run.id, file_id=f.id)
    assert "error" in result
    assert "No vision text artifacts" in result["error"]


def test_payroll_extract_success(session, run):
    f = _make_file(session, run.id)
    _make_artifact(session, f, text="Payroll for Jan 2025")

    llm_response = json.dumps({
        "periods": [
            {
                "period_start": "2025-01-01",
                "period_end": "2025-01-15",
                "total_hours": 320.0,
                "total_wages": 12800.00,
                "currency": "CAD",
                "employee_count": 4,
                "confidence": 0.9,
            }
        ]
    })

    with patch("sred.llm.openai_client.get_chat_completion", return_value=llm_response):
        result = _payroll_extract(session, run.id, file_id=f.id)

    assert result["created"] == 1
    assert result["skipped"] == 0
    assert len(result["periods"]) == 1
    assert result["periods"][0]["status"] == "created"

    # Verify DB record
    pe = session.exec(select(PayrollExtract).where(PayrollExtract.run_id == run.id)).first()
    assert pe is not None
    assert pe.total_hours == 320.0
    assert pe.total_wages == 12800.00
    assert pe.confidence == 0.9


def test_payroll_extract_idempotent(session, run):
    f = _make_file(session, run.id)
    _make_artifact(session, f, text="Payroll for Jan 2025")

    llm_response = json.dumps({
        "periods": [{
            "period_start": "2025-01-01",
            "period_end": "2025-01-15",
            "total_hours": 320.0,
            "confidence": 0.9,
        }]
    })

    with patch("sred.llm.openai_client.get_chat_completion", return_value=llm_response):
        r1 = _payroll_extract(session, run.id, file_id=f.id)
        r2 = _payroll_extract(session, run.id, file_id=f.id)

    assert r1["created"] == 1
    assert r2["created"] == 0
    assert r2["skipped"] == 1


def test_payroll_extract_no_periods(session, run):
    f = _make_file(session, run.id)
    _make_artifact(session, f, text="Not a payroll doc")

    llm_response = json.dumps({"periods": [], "error": "Not a payroll document"})

    with patch("sred.llm.openai_client.get_chat_completion", return_value=llm_response):
        result = _payroll_extract(session, run.id, file_id=f.id)

    assert result["status"] == "no_periods"


# ---------------------------------------------------------------------------
# payroll_validate tests
# ---------------------------------------------------------------------------
def test_payroll_validate_no_extracts(session, run):
    result = _payroll_validate(session, run.id)
    assert "error" in result
    assert "No PayrollExtract" in result["error"]


def test_payroll_validate_no_timesheets(session, run):
    # Create a payroll extract directly
    pe = PayrollExtract(
        run_id=run.id, file_id=1,
        period_start=date(2025, 1, 1), period_end=date(2025, 1, 15),
        total_hours=100.0, confidence=0.9,
    )
    session.add(pe)
    session.commit()

    result = _payroll_validate(session, run.id)
    assert "error" in result
    assert "No TIMESHEET" in result["error"]


def test_payroll_validate_match_within_threshold(session, run):
    """Payroll and timesheet match within 5% — no contradiction."""
    pe = PayrollExtract(
        run_id=run.id, file_id=1,
        period_start=date(2025, 1, 1), period_end=date(2025, 1, 15),
        total_hours=100.0, confidence=0.9,
    )
    session.add(pe)
    session.commit()

    # Create timesheet rows totaling 98h (2% mismatch, under 5%)
    for i in range(7):
        _make_staging_row(session, run.id, "Alice")

    # Override raw_data to have proper dates and hours within the period
    ts_rows = session.exec(
        select(StagingRow).where(StagingRow.run_id == run.id)
    ).all()
    for i, sr in enumerate(ts_rows):
        sr.raw_data = json.dumps({
            "person": "Alice",
            "date": f"2025-01-{(i+1):02d}",
            "hours": 14.0,  # 7 * 14 = 98h
        })
        session.add(sr)
    session.commit()

    result = _payroll_validate(session, run.id)
    assert result["overall_blocking"] is False
    assert result["contradictions_created"] == 0
    assert len(result["period_comparisons"]) == 1
    assert result["period_comparisons"][0]["blocking"] is False


def test_payroll_validate_mismatch_creates_contradiction(session, run):
    """Payroll and timesheet differ by >5% — creates BLOCKING contradiction + task."""
    pe = PayrollExtract(
        run_id=run.id, file_id=1,
        period_start=date(2025, 1, 1), period_end=date(2025, 1, 15),
        total_hours=100.0, confidence=0.9,
    )
    session.add(pe)
    session.commit()

    # Create timesheet rows totaling 50h (50% mismatch)
    for i in range(5):
        raw = json.dumps({
            "person": "Alice",
            "date": f"2025-01-{(i+1):02d}",
            "hours": 10.0,
        })
        sr = StagingRow(
            run_id=run.id,
            raw_data=raw,
            row_type=StagingRowType.TIMESHEET,
            row_hash=hashlib.sha256(raw.encode()).hexdigest(),
            normalized_text=f"Alice 2025-01-{(i+1):02d} 10h",
        )
        session.add(sr)
    session.commit()

    result = _payroll_validate(session, run.id)
    assert result["overall_blocking"] is True
    assert result["contradictions_created"] == 1
    assert result["period_comparisons"][0]["blocking"] is True
    assert result["period_comparisons"][0]["contradiction"] == "created"

    # Verify contradiction in DB
    c = session.exec(
        select(Contradiction).where(
            Contradiction.run_id == run.id,
            Contradiction.contradiction_type == ContradictionType.PAYROLL_MISMATCH,
        )
    ).first()
    assert c is not None
    assert c.severity == ContradictionSeverity.BLOCKING

    # Verify linked ReviewTask
    task = session.exec(
        select(ReviewTask).where(
            ReviewTask.run_id == run.id,
            ReviewTask.issue_key == c.issue_key,
        )
    ).first()
    assert task is not None
    assert task.severity == ContradictionSeverity.BLOCKING
    assert task.contradiction_id == c.id

    # Verify run status changed
    session.refresh(run)
    assert run.status == RunStatus.NEEDS_REVIEW


def test_payroll_validate_dedup_contradiction(session, run):
    """Running validate twice should not create duplicate contradictions."""
    pe = PayrollExtract(
        run_id=run.id, file_id=1,
        period_start=date(2025, 1, 1), period_end=date(2025, 1, 15),
        total_hours=100.0, confidence=0.9,
    )
    session.add(pe)
    session.commit()

    raw = json.dumps({"person": "Alice", "date": "2025-01-05", "hours": 10.0})
    sr = StagingRow(
        run_id=run.id, raw_data=raw,
        row_type=StagingRowType.TIMESHEET,
        row_hash=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_text="Alice 10h",
    )
    session.add(sr)
    session.commit()

    r1 = _payroll_validate(session, run.id)
    r2 = _payroll_validate(session, run.id)

    assert r1["contradictions_created"] == 1
    assert r2["contradictions_created"] == 0
    assert r2["period_comparisons"][0]["contradiction"] == "duplicate"


def test_payroll_validate_no_payroll_hours(session, run):
    """PayrollExtract with total_hours=None should be skipped gracefully."""
    pe = PayrollExtract(
        run_id=run.id, file_id=1,
        period_start=date(2025, 1, 1), period_end=date(2025, 1, 15),
        total_hours=None, total_wages=5000.0, confidence=0.8,
    )
    session.add(pe)
    session.commit()

    _make_staging_row(session, run.id, "Alice")

    result = _payroll_validate(session, run.id)
    assert result["contradictions_created"] == 0
    assert result["period_comparisons"][0]["status"] == "no_payroll_hours"


# ---------------------------------------------------------------------------
# payroll_summary tests
# ---------------------------------------------------------------------------
def test_payroll_summary_empty(session, run):
    result = _payroll_summary(session, run.id)
    assert result["count"] == 0
    assert result["extracts"] == []


def test_payroll_summary_with_data(session, run):
    pe = PayrollExtract(
        run_id=run.id, file_id=1,
        period_start=date(2025, 1, 1), period_end=date(2025, 1, 15),
        total_hours=320.0, total_wages=12800.0, confidence=0.9,
    )
    session.add(pe)
    session.commit()

    result = _payroll_summary(session, run.id)
    assert result["count"] == 1
    assert result["extracts"][0]["total_hours"] == 320.0
    assert result["extracts"][0]["period_start"] == "2025-01-01"
