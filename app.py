import streamlit as st
from pathlib import Path
from rag_engine import load_base, ask_rag

st.set_page_config(
    page_title="SIET Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Sidebar ----------------
st.sidebar.title("🎓 SIET ChatBot")
st.sidebar.caption("Department of Computer Science Engineering")
st.sidebar.divider()

st.sidebar.subheader("📘 About Project")
st.sidebar.write(
    "This chatbot answers questions about the college. "
    "It searches the college PDF files and replies only from the information in them."
)

st.sidebar.subheader("🛠️ Technologies Used")
st.sidebar.markdown("""
- 🐍 Python
- 🎈 Streamlit
- 📄 PyPDF
- 🧠 Sentence Transformers
- 🔎 FAISS Vector Search
- ⚡ Groq LLM
""")

st.sidebar.subheader("🔄 RAG Pipeline")
st.sidebar.graphviz_chart("""
digraph {
    rankdir=TB
    node [shape=box, style="rounded,filled", fillcolor="#E8F0FE", color="#14766E", fontsize=10]
    A [label="PDF Files"]
    B [label="Chunking"]
    C [label="Embeddings"]
    D [label="FAISS Index"]
    E [label="User Question"]
    F [label="Top 3 Chunks"]
    G [label="Groq LLM"]
    H [label="Answer"]
    A -> B -> C -> D
    E -> D [label=" search", fontsize=9]
    D -> F -> G -> H
}
""", use_container_width=True)

# ---------------- Main page ----------------
# Top header row: Title on the left, Clear Chat button on the right
col1, col2 = st.columns([4, 1], vertical_alignment="center")

with col1:
    st.title("🎓 Shridevi Institute of Engineering and Technology - ChatBot")

with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_hist = []
        st.rerun()

st.caption("💡 Ask anything about college information, Courses, Placements, FAQs, Rules and Regulations")
st.caption("⚙️ RAG Pipeline with Multiple Files • Created by Chandrashekar H")
st.divider()

# Safe API key loading
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing in Streamlit Secrets.")
    st.stop()

mykey = st.secrets["GROQ_API_KEY"]

# Resolve folder path safely relative to app.py
DOCS_DIR = Path(__file__).parent / "college_docs"

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

# Chat window icons
user_icon = "🧑‍🎓"
bot_icon = "🤖"

for qa in st.session_state.chat_hist:
    if qa["role"] == "user":
        icon = user_icon
    else:
        icon = bot_icon
    with st.chat_message(qa["role"], avatar=icon):
        st.write(qa["content"])

question = st.chat_input("Ask anything about college....")

if question:
    st.session_state.chat_hist.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=user_icon):
        st.write(question)

    with st.chat_message("assistant", avatar=bot_icon):
        with st.spinner("Searching documents for answer...."):
            try:
                answer = ask_rag(question, embed_model, chunks, index, api_key=mykey)
            except Exception as e:
                answer = f"Sorry, an error occurred while generating the answer: {e}"
            st.write(answer)

    st.session_state.chat_hist.append({"role": "assistant", "content": answer})
