import streamlit as st
import json
from sqlmodel import Session, select, desc
from sred.db import engine
from sred.ui.state import get_run_id
from sred.models.agent_log import ToolCallLog, LLMCallLog

st.title("Agent Runner")

run_id = get_run_id()
if not run_id:
    st.error("Please select a Run first.")
    st.stop()

# --- Run Agent ---
st.subheader("Run Agent")

max_steps = st.slider("Max Steps", min_value=1, max_value=20, value=5)
user_msg = st.text_area("Instruction for the agent", height=120, placeholder="e.g. Process all uploaded files, then profile the CSVs and summarise findings.")
context_notes = st.text_input(
    "Context notes (optional)",
    placeholder="e.g. We are currently resolving identities for File #12",
    help="Injected into the system prompt so the agent knows its immediate goal.",
)

if st.button("Run Agent", type="primary"):
    if not user_msg.strip():
        st.warning("Please enter an instruction.")
    else:
        from sred.agent.runner import run_agent_loop

        with Session(engine) as session:
            with st.spinner("Agent is working..."):
                result = run_agent_loop(
                    session, run_id, user_msg.strip(),
                    max_steps=max_steps,
                    context_notes=context_notes.strip() or None,
                )

        # --- Display trace ---
        st.divider()
        st.subheader("Agent Trace")
        st.caption(f"Completed in {result.total_steps} step(s) — stopped: {result.stopped_reason}")

        for i, step in enumerate(result.steps):
            if step.role == "assistant":
                with st.container(border=True):
                    st.markdown(f"**Step {i + 1} — Assistant**")
                    if step.content:
                        st.markdown(step.content)
                    if step.tool_calls:
                        for tc in step.tool_calls:
                            st.code(f"→ {tc['name']}({tc['arguments']})", language="text")

            elif step.role == "tool":
                with st.container(border=True):
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"**🔧 {step.tool_name}**")
                    cols[1].caption(f"{step.duration_ms} ms")

                    with st.expander("Arguments"):
                        st.json(step.tool_args)
                    with st.expander("Result"):
                        st.json(step.tool_result)

        # Final answer
        if result.final_answer:
            st.divider()
            st.subheader("Final Answer")
            st.markdown(result.final_answer)

st.divider()

# --- Tool Call History ---
st.subheader("Tool Call Log")
with Session(engine) as session:
    logs = session.exec(
        select(ToolCallLog)
        .where(ToolCallLog.run_id == run_id)
        .order_by(desc(ToolCallLog.created_at))
        .limit(50)
    ).all()

    if not logs:
        st.info("No tool calls recorded yet for this run.")
    else:
        for log in logs:
            icon = "✅" if log.success else "❌"
            with st.expander(f"{icon} {log.tool_name} — {log.duration_ms}ms — {log.created_at.strftime('%H:%M:%S')}"):
                st.caption("Arguments")
                try:
                    st.json(log.arguments_json)
                except Exception:
                    st.code(log.arguments_json)
                st.caption("Result")
                try:
                    st.json(log.result_json)
                except Exception:
                    st.code(log.result_json)

# --- LLM Call History ---
st.subheader("LLM Call Log")
with Session(engine) as session:
    llm_logs = session.exec(
        select(LLMCallLog)
        .where(LLMCallLog.run_id == run_id)
        .order_by(desc(LLMCallLog.created_at))
        .limit(20)
    ).all()

    if not llm_logs:
        st.info("No LLM calls recorded yet for this run.")
    else:
        for log in llm_logs:
            with st.expander(f"🤖 {log.model} — {log.total_tokens} tokens — {log.created_at.strftime('%H:%M:%S')}"):
                st.write(f"**Messages:** {log.message_count} | **Tool calls:** {log.tool_calls_count}")
                st.write(f"**Tokens:** prompt={log.prompt_tokens}, completion={log.completion_tokens}")
                st.write(f"**Finish reason:** {log.finish_reason}")
                st.caption("Prompt summary")
                st.text(log.prompt_summary)
