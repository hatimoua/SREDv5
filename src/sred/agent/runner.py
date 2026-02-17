"""
Agent runner: OpenAI tool-calling loop.

Orchestrates a multi-step conversation where the LLM can call registered tools.
Logs every LLM call and tool invocation to the database.
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sqlmodel import Session, select, func
from sred.config import settings
from sred.llm.openai_client import client
from sred.models.agent_log import ToolCallLog, LLMCallLog
from sred.models.core import Run, Person, File, RateStatus, FileStatus
from sred.models.finance import StagingRow, StagingRowType
from sred.models.alias import PersonAlias, AliasStatus
from sred.models.world import Contradiction, ContradictionStatus, ReviewTask, ReviewTaskStatus, DecisionLock
from sred.agent.registry import get_openai_tools_schema, get_tool_handler
from sred.logging import logger

# Ensure all tools are registered on import
import sred.agent.tools  # noqa: F401


SYSTEM_PROMPT = """\
You are the SR&ED Automation Agent. You help Canadian companies prepare SR&ED tax credit claims.

You have access to tools for:
- Ingesting and processing uploaded files (PDF, CSV, DOCX, images)
- Searching extracted text segments
- Profiling and querying CSV data via DuckDB
- Managing people (employees/contractors)
- Resolving person-name aliases (entity resolution)
- Creating and listing tasks (hypotheses)
- Flagging contradictions for human review
- Checking locked decisions
- Writing memory/summary documents

Rules:
- Never write raw SQL or modify the database directly. Always use the provided tools.
- Be concise and action-oriented.
- When uncertain, create a task or contradiction for human review rather than guessing.
- Explain your reasoning briefly before each tool call.
"""


def build_run_context(session: Session, run_id: int) -> str:
    """Build a dynamic context block describing the current run state.

    Returns a concise multi-line string suitable for injection into the
    system prompt so the agent knows its immediate situation.
    """
    run = session.get(Run, run_id)
    if not run:
        return ""

    lines: list[str] = [f"Run: #{run.id} \"{run.name}\" — status: {run.status.value}"]

    # People
    people = session.exec(select(Person).where(Person.run_id == run_id)).all()
    pending_rates = [p for p in people if p.rate_status == RateStatus.PENDING]
    lines.append(f"People: {len(people)} total, {len(pending_rates)} with PENDING rate")

    # Files
    files = session.exec(select(File).where(File.run_id == run_id)).all()
    processed = sum(1 for f in files if f.status == FileStatus.PROCESSED)
    lines.append(f"Files: {len(files)} uploaded, {processed} processed")

    # Staging rows
    ts_count = session.exec(
        select(func.count(StagingRow.id)).where(
            StagingRow.run_id == run_id,
            StagingRow.row_type == StagingRowType.TIMESHEET,
        )
    ).one()
    lines.append(f"Timesheet staging rows: {ts_count}")

    # Aliases
    alias_confirmed = session.exec(
        select(func.count(PersonAlias.id)).where(
            PersonAlias.run_id == run_id,
            PersonAlias.status == AliasStatus.CONFIRMED,
        )
    ).one()
    alias_total = session.exec(
        select(func.count(PersonAlias.id)).where(PersonAlias.run_id == run_id)
    ).one()
    lines.append(f"Person aliases: {alias_confirmed} confirmed / {alias_total} total")

    # Open contradictions & tasks
    open_contradictions = session.exec(
        select(func.count(Contradiction.id)).where(
            Contradiction.run_id == run_id,
            Contradiction.status == ContradictionStatus.OPEN,
        )
    ).one()
    open_tasks = session.exec(
        select(func.count(ReviewTask.id)).where(
            ReviewTask.run_id == run_id,
            ReviewTask.status == ReviewTaskStatus.OPEN,
        )
    ).one()
    lines.append(f"Open contradictions: {open_contradictions}, open tasks: {open_tasks}")

    # Active locks
    active_locks = session.exec(
        select(func.count(DecisionLock.id)).where(
            DecisionLock.run_id == run_id,
            DecisionLock.active == True,  # noqa: E712
        )
    ).one()
    lines.append(f"Active decision locks: {active_locks}")

    return "\n".join(lines)


@dataclass
class AgentStep:
    """One step in the agent trace."""
    role: str  # "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


@dataclass
class AgentResult:
    """Final result of an agent run."""
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    stopped_reason: str = ""  # "complete", "max_steps", "error"


def _log_llm_call(session: Session, run_id: int, model: str, messages: list, response, session_id: str | None = None) -> None:
    """Persist an LLM call summary to the database."""
    usage = response.usage
    tool_calls = response.choices[0].message.tool_calls or []

    # Summarise: last user/assistant message (truncated)
    prompt_summary = ""
    for m in reversed(messages):
        if m["role"] in ("user", "system"):
            prompt_summary = str(m.get("content", ""))[:500]
            break

    log = LLMCallLog(
        run_id=run_id,
        session_id=session_id,
        model=model,
        prompt_summary=prompt_summary,
        message_count=len(messages),
        tool_calls_count=len(tool_calls),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        finish_reason=response.choices[0].finish_reason,
    )
    session.add(log)
    session.commit()


def _log_tool_call(session: Session, run_id: int, tool_name: str, args_json: str, result: dict, success: bool, duration_ms: int, session_id: str | None = None) -> None:
    """Persist a tool call to the database."""
    log = ToolCallLog(
        run_id=run_id,
        session_id=session_id,
        tool_name=tool_name,
        arguments_json=args_json,
        result_json=json.dumps(result, default=str)[:4000],
        success=success,
        duration_ms=duration_ms,
    )
    session.add(log)
    session.commit()


def run_agent_loop(
    session: Session,
    run_id: int,
    user_message: str,
    max_steps: int = 10,
    context_notes: str | None = None,
) -> AgentResult:
    """
    Execute the agent loop.

    1. Send user message + tool schemas to OpenAI.
    2. If the model returns tool_calls, execute them and feed results back.
    3. Repeat until the model returns a plain text response or max_steps reached.

    Args:
        context_notes: Optional caller-supplied context (e.g. "We are currently
            resolving identities for File #12"). Injected into the system prompt
            alongside the auto-generated run snapshot.
    """
    result = AgentResult()
    model = settings.OPENAI_MODEL_AGENT
    tools_schema = get_openai_tools_schema()
    session_id = str(uuid.uuid4())

    # --- Dynamic system prompt ---
    prompt_parts = [SYSTEM_PROMPT]

    run_ctx = build_run_context(session, run_id)
    if run_ctx:
        prompt_parts.append(f"## Current Run State\n{run_ctx}")

    if context_notes:
        prompt_parts.append(f"## Immediate Goal\n{context_notes}")

    system_content = "\n\n".join(prompt_parts)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]

    for step_num in range(max_steps):
        logger.info(f"Agent step {step_num + 1}/{max_steps}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
            )
        except Exception as e:
            logger.error(f"OpenAI API error at step {step_num + 1}: {e}")
            result.stopped_reason = f"error: {e}"
            break

        # Log the LLM call
        _log_llm_call(session, run_id, model, messages, response, session_id=session_id)

        choice = response.choices[0]
        assistant_msg = choice.message

        # If no tool calls, we have the final answer
        if not assistant_msg.tool_calls:
            final_text = assistant_msg.content or ""
            result.steps.append(AgentStep(role="assistant", content=final_text))
            result.final_answer = final_text
            result.stopped_reason = "complete"
            result.total_steps = step_num + 1
            return result

        # Record assistant step with tool calls
        tc_summaries = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in assistant_msg.tool_calls
        ]
        result.steps.append(AgentStep(
            role="assistant",
            content=assistant_msg.content,
            tool_calls=tc_summaries,
        ))

        # Append assistant message to conversation
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ],
        })

        # Execute each tool call
        for tc in assistant_msg.tool_calls:
            tool_name = tc.function.name
            args_json = tc.function.arguments

            try:
                kwargs = json.loads(args_json)
            except json.JSONDecodeError:
                kwargs = {}

            t0 = time.monotonic()
            success = True
            try:
                handler = get_tool_handler(tool_name)
                tool_result = handler(session, run_id, **kwargs)
            except KeyError:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
                success = False
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                tool_result = {"error": str(e)}
                success = False

            duration_ms = int((time.monotonic() - t0) * 1000)

            # Log tool call
            _log_tool_call(session, run_id, tool_name, args_json, tool_result, success, duration_ms, session_id=session_id)

            # Record step
            result.steps.append(AgentStep(
                role="tool",
                tool_name=tool_name,
                tool_args=kwargs,
                tool_result=tool_result,
                duration_ms=duration_ms,
            ))

            # Append tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, default=str),
            })

    else:
        # Exhausted max_steps
        result.stopped_reason = "max_steps"
        result.final_answer = "(Agent reached maximum steps without a final answer.)"

    result.total_steps = len([s for s in result.steps if s.role == "assistant"])
    return result
