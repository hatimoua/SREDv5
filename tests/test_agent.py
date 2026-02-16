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
from sred.agent.tools import (
    _people_list,
    _people_get,
    _tasks_list_open,
    _tasks_create,
    _contradictions_list_open,
    _contradictions_create,
    _locks_list_active,
    _memory_write_summary,
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
    }
    assert expected.issubset(set(TOOL_REGISTRY.keys()))


def test_openai_schema_format():
    schema = get_openai_tools_schema()
    assert len(schema) >= 12
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
