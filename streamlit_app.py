import os
import io
import nbformat
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr

# --- 0) Secrets → env (no keys leak to users) ---
# Add any other keys you use in the notebook here:
for k in ("GROQ_API_KEY",):
    if k in st.secrets:
        os.environ[k] = str(st.secrets[k])

# Optional: point your Redfin CSV to a repo file instead of Google Drive.
# If your notebook hardcodes /content/drive/... this override is used by the loader below.
DEFAULT_REDFIN = os.environ.get("REDFIN_CSV_PATH", "data/redfin.csv")

# --- 1) Execute notebook into a shared globals namespace once ---
@st.cache_resource(show_spinner=True)  # runs once per server session
def bootstrap_notebook(nb_path: str):
    g = {
        "__name__": "__main__",
        "file_path": DEFAULT_REDFIN,  # many notebooks read this var—gives us a hook
    }

    nb = nbformat.read(nb_path, as_version=4)

    # Small patcher: if a cell sets file_path to /content/drive... swap for env-driven path.
    def maybe_patch(src: str) -> str:
        if "file_path =" in src and "redfin.csv" in src and "/content/drive" in src:
            return (
                "import os\n"
                "file_path = os.environ.get('REDFIN_CSV_PATH', 'data/redfin.csv')\n"
            )
        return src

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            src = cell.source or ""
            # Skip magics and interactive input loops if any (keeps the server non-blocking)
            if src.strip().startswith("%") or "input(" in src:
                continue
            exec(maybe_patch(src), g)

    logs = stdout_buf.getvalue() + "\n" + stderr_buf.getvalue()
    return g, logs

g, boot_logs = bootstrap_notebook("RI_UsingCustomTools_GroqVersion_workingVersion.ipynb")

# Pull compiled graph (LangGraph) and message classes from your notebook's globals
app = g.get("app", None)

# (Optional) try to import message classes for better type fidelity
try:
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
except Exception:
    HumanMessage = AIMessage = BaseMessage = None

st.set_page_config(page_title="GRACIE – AI Realtor", page_icon="🏠", layout="wide")
st.title("🏠 GRACIE – AI Realtor (Notebook-powered)")

with st.expander("Server boot logs", expanded=False):
    st.code(boot_logs or "(no logs)")

if app is None:
    st.error("Couldn’t find the compiled graph `app` in the notebook. Make sure the notebook defines and compiles `app`.")
    st.stop()

# --- 2) Simple chat UI using Streamlit's chat components ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optional system intro:
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I’m GRACIE. Ask me about properties, zip codes, CMAs, or NPV."})

# Render history
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"]=="assistant" else "user"):
        st.markdown(m["content"])

user_input = st.chat_input("Type your question…")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare LangChain-style message list if available
    lc_messages = []
    if HumanMessage and AIMessage:
        for m in st.session_state.messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
    else:
        # Fallback: the graph often accepts dicts with "content"
        lc_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # Call your compiled graph
    try:
        # Most LangGraph apps accept {"messages": [...]} and return a dict with "messages"
        result = app.invoke({"messages": lc_messages})

        # Extract latest assistant message
        assistant_text = None

        if isinstance(result, dict) and "messages" in result:
            msgs = result["messages"]
            if msgs:
                last = msgs[-1]
                # Handle both LangChain BaseMessage and dict
                if hasattr(last, "content"):
                    assistant_text = last.content
                elif isinstance(last, dict):
                    assistant_text = last.get("content")
        # Fallback: some graphs return a string
        if not assistant_text and isinstance(result, str):
            assistant_text = result

        assistant_text = assistant_text or "✅ (No content returned by the graph.)"
    except Exception as e:
        assistant_text = f"⚠️ Error while running the agent: `{e}`"

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
