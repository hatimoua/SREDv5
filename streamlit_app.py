import streamlit as st
from sred.ui.validation import run_all_checks
from sred.ui.state import init_session

# Page configuration
st.set_page_config(
    page_title="SR&ED Automation PoC",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Run pre-flight checks
errors = run_all_checks()

if errors:
    st.error("🚨 System Configuration Errors")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

# Initialize State
init_session()

# Navigation
st.sidebar.title("SR&ED PoC")

# Multipage definition
pg = st.navigation([
    st.Page("src/sred/ui/pages/1_run.py", title="Runs", icon="🚀"),
    st.Page("src/sred/ui/pages/2_people.py", title="People", icon="👥"),
    st.Page("src/sred/ui/pages/3_uploads.py", title="Uploads", icon="📂"),
    st.Page("src/sred/ui/pages/4_dashboard.py", title="Dashboard", icon="📊"),
    st.Page("src/sred/ui/pages/7_agent.py", title="Agent Runner", icon="🤖"),
    st.Page("src/sred/ui/pages/8_tasks.py", title="Tasks & Gates", icon="🚦"),
    st.Page("src/sred/ui/pages/9_payroll.py", title="Payroll Validation", icon="💰"),
    st.Page("src/sred/ui/pages/10_trace.py", title="Execution Trace", icon="🔍"),
    st.Page("src/sred/ui/pages/11_ledger.py", title="Labour Ledger", icon="📒"),
])

pg.run()
