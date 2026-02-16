"""
Agent runner: OpenAI tool-calling loop.

Orchestrates a multi-step conversation where the LLM can call registered tools.
Logs every LLM call and tool invocation to the database.
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sqlmodel import Session
from sred.config import settings
from sred.llm.openai_client import client
from sred.models.agent_log import ToolCallLog, LLMCallLog
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


def _log_llm_call(session: Session, run_id: int, model: str, messages: list, response) -> None:
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


def _log_tool_call(session: Session, run_id: int, tool_name: str, args_json: str, result: dict, success: bool, duration_ms: int) -> None:
    """Persist a tool call to the database."""
    log = ToolCallLog(
        run_id=run_id,
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
) -> AgentResult:
    """
    Execute the agent loop.

    1. Send user message + tool schemas to OpenAI.
    2. If the model returns tool_calls, execute them and feed results back.
    3. Repeat until the model returns a plain text response or max_steps reached.
    """
    result = AgentResult()
    model = settings.OPENAI_MODEL_AGENT
    tools_schema = get_openai_tools_schema()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
        _log_llm_call(session, run_id, model, messages, response)

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
            _log_tool_call(session, run_id, tool_name, args_json, tool_result, success, duration_ms)

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
