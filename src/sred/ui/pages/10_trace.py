"""
Execution Trace Inspector — browse past agent sessions.

Groups ToolCallLog and LLMCallLog entries by session_id to reconstruct
the full step-by-step trace of each agent loop invocation.
"""
import streamlit as st
import json
from datetime import datetime
from sqlmodel import Session, select, desc, col
from sred.db import engine
from sred.ui.state import get_run_id
from sred.models.agent_log import ToolCallLog, LLMCallLog

st.title("Execution Trace Inspector")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()

# ---------------------------------------------------------------------------
# 1. Discover all sessions for this run
# ---------------------------------------------------------------------------
with Session(engine) as db:
    # Get distinct session_ids from LLMCallLog (one LLM call per agent step)
    llm_rows = db.exec(
        select(LLMCallLog)
        .where(LLMCallLog.run_id == run_id, LLMCallLog.session_id != None)  # noqa: E711
        .order_by(desc(LLMCallLog.created_at))
    ).all()

    # Group by session_id, preserving order (most recent first)
    seen_sessions: dict[str, dict] = {}
    for row in llm_rows:
        sid = row.session_id
        if sid and sid not in seen_sessions:
            seen_sessions[sid] = {
                "session_id": sid,
                "started_at": row.created_at,
                "model": row.model,
                "first_prompt": row.prompt_summary[:120] if row.prompt_summary else "",
            }

    sessions = list(seen_sessions.values())

if not sessions:
    st.info("No agent sessions recorded yet for this run. Run the agent first on the Agent Runner page.")
    st.stop()

# ---------------------------------------------------------------------------
# 2. Session selector
# ---------------------------------------------------------------------------
st.subheader(f"📋 {len(sessions)} Agent Session(s)")

session_labels = [
    f"{s['started_at'].strftime('%Y-%m-%d %H:%M:%S')} — {s['model']} — \"{s['first_prompt']}…\""
    for s in sessions
]

selected_idx = st.selectbox(
    "Select a session to inspect",
    range(len(sessions)),
    format_func=lambda i: session_labels[i],
)

selected = sessions[selected_idx]
sid = selected["session_id"]

st.caption(f"Session ID: `{sid}`")

# ---------------------------------------------------------------------------
# 3. Load full trace for selected session
# ---------------------------------------------------------------------------
with Session(engine) as db:
    llm_calls = db.exec(
        select(LLMCallLog)
        .where(LLMCallLog.run_id == run_id, LLMCallLog.session_id == sid)
        .order_by(LLMCallLog.created_at)
    ).all()

    tool_calls = db.exec(
        select(ToolCallLog)
        .where(ToolCallLog.run_id == run_id, ToolCallLog.session_id == sid)
        .order_by(ToolCallLog.created_at)
    ).all()

# ---------------------------------------------------------------------------
# 4. Build interleaved event timeline (used by both download and display)
# ---------------------------------------------------------------------------
events: list[tuple[datetime, str, object]] = []
for l in llm_calls:
    events.append((l.created_at, "llm", l))
for t in tool_calls:
    events.append((t.created_at, "tool", t))
events.sort(key=lambda e: e[0])

# ---------------------------------------------------------------------------
# 5. Build Markdown export & download button
# ---------------------------------------------------------------------------

def _build_trace_md() -> str:
    """Render the full session trace as a Markdown document."""
    lines: list[str] = []
    lines.append(f"# Execution Trace — Session {sid[:8]}")
    lines.append(f"")
    lines.append(f"- **Run ID:** {run_id}")
    lines.append(f"- **Session ID:** `{sid}`")
    lines.append(f"- **Started:** {selected['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Model:** {selected['model']}")
    lines.append("")

    # Summary
    lines.append("## Session Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| LLM Calls | {len(llm_calls)} |")
    lines.append(f"| Tool Calls | {len(tool_calls)} |")
    lines.append(f"| Total Tokens | {sum(l.total_tokens for l in llm_calls):,} |")
    lines.append(f"| Tool Execution Time | {sum(t.duration_ms for t in tool_calls):,} ms |")
    lines.append(f"| Failed Tools | {sum(1 for t in tool_calls if not t.success)} |")
    lines.append("")

    # Step-by-step
    lines.append("## Step-by-Step Trace")
    lines.append("")
    step = 0
    for ts, kind, obj in events:
        if kind == "llm":
            step += 1
            l: LLMCallLog = obj  # type: ignore
            lines.append(f"### Step {step} — LLM Call (`{l.model}`) — {ts.strftime('%H:%M:%S')}")
            lines.append("")
            lines.append(f"- Messages: {l.message_count}")
            lines.append(f"- Tool calls requested: {l.tool_calls_count}")
            lines.append(f"- Tokens: {l.total_tokens} (prompt: {l.prompt_tokens}, completion: {l.completion_tokens})")
            lines.append(f"- Finish reason: {l.finish_reason}")
            lines.append("")
            if l.prompt_summary:
                lines.append("**Prompt Summary:**")
                lines.append("```")
                lines.append(l.prompt_summary)
                lines.append("```")
                lines.append("")
        elif kind == "tool":
            t: ToolCallLog = obj  # type: ignore
            status = "SUCCESS" if t.success else "FAILED"
            lines.append(f"#### Tool: `{t.tool_name}` — {status} — {t.duration_ms} ms — {ts.strftime('%H:%M:%S')}")
            lines.append("")
            lines.append("**Arguments:**")
            lines.append("```json")
            try:
                lines.append(json.dumps(json.loads(t.arguments_json), indent=2))
            except Exception:
                lines.append(t.arguments_json)
            lines.append("```")
            lines.append("")
            lines.append("**Result:**")
            lines.append("```json")
            try:
                lines.append(json.dumps(json.loads(t.result_json), indent=2))
            except Exception:
                lines.append(t.result_json)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


md_content = _build_trace_md()
filename = f"trace_{sid[:8]}_{selected['started_at'].strftime('%Y%m%d_%H%M%S')}.md"

st.download_button(
    label="Download Trace as Markdown",
    data=md_content,
    file_name=filename,
    mime="text/markdown",
    icon="📥",
)

# ---------------------------------------------------------------------------
# 5. Summary metrics
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Session Summary")

total_tokens = sum(l.total_tokens for l in llm_calls)
total_tool_time = sum(t.duration_ms for t in tool_calls)
failed_tools = sum(1 for t in tool_calls if not t.success)
finish_reasons = [l.finish_reason or "?" for l in llm_calls]

cols = st.columns(5)
cols[0].metric("LLM Calls", len(llm_calls))
cols[1].metric("Tool Calls", len(tool_calls))
cols[2].metric("Total Tokens", f"{total_tokens:,}")
cols[3].metric("Tool Time", f"{total_tool_time:,} ms")
cols[4].metric("Failed Tools", failed_tools)

# ---------------------------------------------------------------------------
# 7. Interleaved timeline
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Step-by-Step Trace")

step_num = 0
for ts, kind, obj in events:
    if kind == "llm":
        step_num += 1
        llm: LLMCallLog = obj  # type: ignore
        with st.container(border=True):
            header_cols = st.columns([4, 1])
            header_cols[0].markdown(f"**Step {step_num} — 🤖 LLM Call** (`{llm.model}`)")
            header_cols[1].caption(ts.strftime("%H:%M:%S"))

            mcols = st.columns(4)
            mcols[0].write(f"Messages: **{llm.message_count}**")
            mcols[1].write(f"Tool calls: **{llm.tool_calls_count}**")
            mcols[2].write(f"Tokens: **{llm.total_tokens}** (p:{llm.prompt_tokens} c:{llm.completion_tokens})")
            mcols[3].write(f"Finish: **{llm.finish_reason}**")

            with st.expander("Prompt Summary"):
                st.text(llm.prompt_summary or "(empty)")

    elif kind == "tool":
        tc: ToolCallLog = obj  # type: ignore
        icon = "✅" if tc.success else "❌"
        with st.container(border=True):
            header_cols = st.columns([4, 1, 1])
            header_cols[0].markdown(f"**{icon} 🔧 {tc.tool_name}**")
            header_cols[1].caption(f"{tc.duration_ms} ms")
            header_cols[2].caption(ts.strftime("%H:%M:%S"))

            with st.expander("Arguments"):
                try:
                    st.json(tc.arguments_json)
                except Exception:
                    st.code(tc.arguments_json)

            with st.expander("Result"):
                try:
                    st.json(tc.result_json)
                except Exception:
                    st.code(tc.result_json)

# ---------------------------------------------------------------------------
# 6. Raw data tables (collapsed)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📊 Raw LLM Call Data"):
    if llm_calls:
        st.dataframe(
            [
                {
                    "Time": l.created_at.strftime("%H:%M:%S"),
                    "Model": l.model,
                    "Messages": l.message_count,
                    "Tool Calls": l.tool_calls_count,
                    "Prompt Tokens": l.prompt_tokens,
                    "Completion Tokens": l.completion_tokens,
                    "Total Tokens": l.total_tokens,
                    "Finish": l.finish_reason,
                }
                for l in llm_calls
            ],
            use_container_width=True,
        )

with st.expander("📊 Raw Tool Call Data"):
    if tool_calls:
        st.dataframe(
            [
                {
                    "Time": t.created_at.strftime("%H:%M:%S"),
                    "Tool": t.tool_name,
                    "Success": "✅" if t.success else "❌",
                    "Duration (ms)": t.duration_ms,
                    "Args": t.arguments_json[:80] + "…" if len(t.arguments_json) > 80 else t.arguments_json,
                    "Result": t.result_json[:80] + "…" if len(t.result_json) > 80 else t.result_json,
                }
                for t in tool_calls
            ],
            use_container_width=True,
        )
