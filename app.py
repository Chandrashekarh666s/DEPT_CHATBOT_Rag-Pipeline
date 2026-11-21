import streamlit as st
from rag_engine import load_base, ask_rag

st.set_page_config(page_title="SIET_Chatbot", page_icon="🎓", layout="centered")

with st.sidebar:
    st.title("🎓 SIET AI ChatBot")
    st.caption("Shridevi Institute of Engineering and Technology, Tumkur")
    st.divider()

    st.subheader("📄 About Project")
    st.write("A college assistant built using RAG to answer questions about courses, faculty profiles, calendars, placements, and college rules from PDF files.")

    st.subheader("⚙️ Technologies Used")
    st.markdown("""
    - **Frontend:** Streamlit
    - **PDF Reader:** PyPDF
    - **Embedding:** Sentence Transformers
    - **Vector DB:** FAISS
    - **LLM:** Groq API
    """)

    st.subheader("🔄 RAG Pipeline Flow")
    st.code("""
Loading (PDF/txt) [PDF Reader]
        ↓
Chunking
        ↓
Embedding [Sentence Transformer]
        ↓
Vector DB [FAISS]
        ↓
User Queries [Embedding]
        ↓
Retrieval [FAISS]
        ↓
LLM [Chunks are given]
        ↓
Final Answer
    """, language="text")

    st.divider()
    btn = st.button("🗑️ Clear Chat", use_container_width=True)
    if btn:
        st.session_state.chat_hist = []
        st.success("Chat History Deleted")
        st.rerun()

    st.caption("Created by Chandrashekar H")

st.title("Shridevi Institute of Engineering and Technology - ChatBot")
st.caption("Ask anything about college information, Courses, Placements, FAQs, Rules and Regulations")
st.caption("RAG pipeline with Multiple Files")

mykey = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def initiate_rag():
    return load_base("college_docs")

index, embed_model, chunks = initiate_rag()

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])

question = st.chat_input("Ask anything about college....")

if question:
    st.session_state.chat_hist.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Asking RAG for answer...."):
            answer = ask_rag(question, embed_model, chunks, index, api_key=mykey)
            st.write(answer)

            st.session_state.chat_hist.append({
                "role": "assistant",
                "content": answer
            })
