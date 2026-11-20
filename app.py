import streamlit as st
from pathlib import Path
from rag_engine import load_base, ask_rag

st.set_page_config(
    page_title="SIET Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Top header row: Title on the left, Clear Chat button on the right
col1, col2 = st.columns([4, 1], vertical_alignment="center")

with col1:
    st.title("Shridevi Institute of Engineering and Technology - ChatBot")

with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_hist = []
        st.rerun()

st.caption("Department of Computer Science Engineering")
st.caption("RAG Pipeline with Multiple Files • Created by Chandrashekar H")
st.divider()

# Safe API key loading
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing in Streamlit Secrets.")
    st.stop()

mykey = st.secrets["GROQ_API_KEY"]

# Resolve folder path safely relative to app.py
DOCS_DIR = Path(__file__).parent / "class_docs"

@st.cache_resource(show_spinner="Loading college knowledge base...")
def initiate_rag():
    return load_base(str(DOCS_DIR))

try:
    index, embed_model, chunks = initiate_rag()
except Exception as e:
    st.error(f"Failed to load documents: {e}")
    st.stop()

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

if "page" not in st.session_state:
    st.session_state.page = "Chat"

if "tasks" not in st.session_state:
    st.session_state.tasks = []

# Sidebar buttons
st.sidebar.title("🎓 SIET Bot")
if st.sidebar.button("💬 Chat", use_container_width=True):
    st.session_state.page = "Chat"
if st.sidebar.button("📅 Planner", use_container_width=True):
    st.session_state.page = "Planner"
if st.sidebar.button("ℹ️ About", use_container_width=True):
    st.session_state.page = "About"

# ---------------- Chat page ----------------
if st.session_state.page == "Chat":

    for qa in st.session_state.chat_hist:
        with st.chat_message(qa["role"]):
            st.write(qa["content"])

    question = st.chat_input("Ask anything about college....")

    # Sample question buttons, shown only before the first message
    if len(st.session_state.chat_hist) == 0:
        if st.button("What are the placement details?"):
            question = "What are the placement details?"
        if st.button("Which courses does the CSE department offer?"):
            question = "Which courses does the CSE department offer?"
        if st.button("What are the rules and regulations?"):
            question = "What are the rules and regulations?"

    if question:
        st.session_state.chat_hist.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents for answer...."):
                try:
                    answer = ask_rag(question, embed_model, chunks, index, api_key=mykey)
                except Exception as e:
                    answer = f"Sorry, an error occurred while generating the answer: {e}"
                st.write(answer)

        st.session_state.chat_hist.append({"role": "assistant", "content": answer})

# ---------------- Planner page ----------------
elif st.session_state.page == "Planner":
    st.subheader("📅 Planner")

    # Pick lines from the college_docs files that contain the selected topic
    topic = st.selectbox("Topic", ["exam", "placement", "fee", "event", "deadline"])

    for file in DOCS_DIR.glob("*.txt"):
        for line in file.read_text(encoding="utf-8").splitlines():
            if topic in line.lower():
                st.write("•", line)

    st.divider()
    st.write("My Tasks")
    new_task = st.text_input("Add a task")
    if st.button("Add") and new_task:
        st.session_state.tasks.append(new_task)
        st.rerun()

    for i, task in enumerate(st.session_state.tasks):
        if st.button("✅ " + task, key=i):
            st.session_state.tasks.pop(i)
            st.rerun()

# ---------------- About page ----------------
else:
    st.subheader("ℹ️ About")
    st.write("Shridevi Institute of Engineering and Technology - CSE Department Chatbot")
    st.write("Technologies: Python, Streamlit, FAISS, Sentence Embeddings, Groq LLM")
