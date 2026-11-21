import re
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    mypages = [i.extract_text() or "" for i in reader.pages]
    full_text = "\n".join(mypages).strip()
    if not full_text:
        raise ValueError("No readable Text Found")
    return full_text

def create_chunks(full_text, chunksize=900, overlap=150):
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def split_records(full_text):
    parts = re.split(
        r"(?m)(?=(?:^|\|\s*)(?:PROFILE:\s*|# [^\n]*(?:Profile|Calendar)|[^\n|]*Semester Calendar[^\n]*$|Faculty Profiles$))",
        full_text
    )
    return [p.strip() for p in parts if p.strip()]

def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot apply embedding without chunks")
    embed_model = SentenceTransformer(model_name)
    embed_vectors = embed_model.encode(chunks, convert_to_numpy=True)
    embed_vectors = np.asarray(embed_vectors, dtype="float32")
    faiss.normalize_L2(embed_vectors)
    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)
    return index, embed_model

def load_base(pdf_path="college_docs"):
    chunks = []
    seen = set()
    for pdf in sorted(Path(pdf_path).glob("*")):
        if pdf.suffix.lower() not in [".pdf", ".md", ".txt"]:
            continue
        try:
            if pdf.suffix.lower() == ".pdf":
                full_text = load_pdf(pdf)
            else:
                full_text = pdf.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            print("Skipped (no readable text):", pdf.name)
            continue

        for record in split_records(full_text):
            raw_title = record.splitlines()[0].strip()
            title = re.sub(r"^[\|\s]+|[\|\s]+$", "", raw_title)[:80]
            for c in create_chunks(record):
                chunk = f"[{title}]\n{c}"
                if chunk not in seen:
                    seen.add(chunk)
                    chunks.append(chunk)
        print("Loaded:", pdf.name, "| total chunks so far:", len(chunks))

    if not chunks:
        raise ValueError("No readable PDF files found in " + str(pdf_path))
    index, embed_model = build_vectors(chunks)
    return index, embed_model, chunks

def retrieve_content(question, embed_model, chunks, index, num_chunks=6):
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if num_chunks < 1:
        raise ValueError("Request atleast one or more chunks")

    query_embed = embed_model.encode([question], convert_to_numpy=True)
    query_embed = np.asarray(query_embed, dtype="float32")
    faiss.normalize_L2(query_embed)
    sim_scores, vec_indices = index.search(query_embed, min(num_chunks, len(chunks)))

    selected_indices = list(vec_indices[0])

    # Ignore common English words so keyword search locks onto actual names/terms like "suthan"
    stopwords = {"what", "when", "where", "which", "whom", "whose", "tell", "about", "give", "show", "details", "information", "profile", "faculty", "professor", "college", "does", "have", "from", "with", "that", "this", "there", "their"}
    q_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", question) if w.lower() not in stopwords]

    for idx, ch in enumerate(chunks):
        if idx not in selected_indices and any(w in ch.lower() for w in q_words):
            selected_indices.append(idx)
            if len(selected_indices) >= num_chunks + 4:
                break

    background = ""
    for i in selected_indices:
        if i != -1:
            background += chunks[i] + "\n\n"
    return background

def ask_rag(question, embed_model, chunks, index, api_key, num_chunks=6, model_name="openai/gpt-oss-120b"):
    if not api_key:
        raise ValueError("API Key Missing")
    background = retrieve_content(question, embed_model, chunks, index, num_chunks=num_chunks)
    prompt = (
        "You are a professional AI Assistant.\n"
        "Answer the user question using ONLY the document background below.\n"
        "Do not invent or assume facts.\n"
        "If information is not available, simply say Information Missing.\n\n"
        f"Document Background:\n{background}\n"
        f"User Question: {question}\n"
        "Answer:"
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
