import re
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

#load pdfs
def load_pdf(pdf_path):
  reader = PdfReader(str(pdf_path))
  mypages = [i.extract_text()  or "" for i in reader.pages]
  full_text = "\n".join(mypages).strip()
  if not full_text:
    raise ValueError("No readable Text Found")
  return full_text

#Chunking
def create_chunks(full_text, chunksize = 500, overlap=100):
  chunks = []
  start = 0
  while start<len(full_text):
    end = start + chunksize
    chunk = full_text[start:end]
    if chunk.strip():
      chunks.append(chunk)
    start = end - overlap
  return chunks

#Records: one profile / one calendar per record (FIX 7)
def split_records(full_text):
  parts = re.split(r"(?m)^(?=PROFILE: |# [^\n]*(?:Profile|Calendar)|[^\n|]*Semester Calendar[^\n|]*$|Faculty Profiles$)", full_text)
  return [p.strip() for p in parts if p.strip()]

#Embedding:
def build_vectors (chunks, model_name = "all-MiniLM-L6-v2"):
  if not chunks:
    raise ValueError("Cannot apply embedding without chunks")
  embed_model = SentenceTransformer(model_name)
  embed_vectors = embed_model.encode(chunks, convert_to_numpy = True)
  faiss.normalize_L2(embed_vectors)
  index = faiss.IndexFlatIP(embed_vectors.shape[1])
  index.add(embed_vectors)
  return index,embed_model

#call loadpdf,chunk,embedding
def load_base (pdf_path= "college_docs"):
  chunks=[]
  seen=set()
  # FIX 1: match .pdf / .PDF both (Linux is case sensitive)
  for pdf in sorted(Path(pdf_path).glob("*")):
    # FIX 6: also read .md and .txt files, not only pdf
    if pdf.suffix.lower() not in [".pdf", ".md", ".txt"]:
      continue
    # FIX 2: one unreadable (scanned) pdf should not crash the whole app
    try:
      if pdf.suffix.lower() == ".pdf":
        full_text = load_pdf(pdf)
      else:
        full_text = pdf.read_text(encoding="utf-8", errors="ignore")
    except Exception:
      print("Skipped (no readable text):", pdf.name)
      continue
    # FIX 7: split into one record per profile / calendar, then put the record
    # title on every chunk, so a chunk always knows whose data it is.
    # Repeated chunks (same text in the combined file and the single file) are added only once.
    for record in split_records(full_text):
      title = record.splitlines()[0].strip()[:80]
      for c in create_chunks(record):
        chunk = title + "\n" + c
        if chunk not in seen:
          seen.add(chunk)
          chunks.append(chunk)
    print("Loaded:", pdf.name, "| total chunks so far:", len(chunks))
  # FIX 3: clear error when nothing was loaded
  if not chunks:
    raise ValueError("No readable PDF files found in " + str(pdf_path))
  index,  embed_model = build_vectors(chunks)
  return  index, embed_model, chunks

#retrieve
def retrieve_content (question, embed_model, chunks, index, num_chunks=3):
  if not question.strip():
    raise ValueError("Question cannot be blank")
  if num_chunks < 1:
    raise ValueError("Request atleast one or more chunks")
  query_embed = embed_model.encode([question], convert_to_numpy=True)
  faiss.normalize_L2(query_embed)
  sim_scores, vec_indices = index.search(query_embed, min(num_chunks, len(chunks)))
  background = ""
  for i in vec_indices[0]:
    if i != -1:
      background += chunks[i] + "\n\n"
  return background

#Ask:
def ask_rag (question, embed_model, chunks, index, api_key, num_chunks=3, model_name = "openai/gpt-oss-120b"):
  key = api_key
  if not key:
    raise ValueError("API Key Missing")
  # FIX 4: num_chunks was hard coded as 3, now the argument is used
  background = retrieve_content (question, embed_model, chunks, index, num_chunks=num_chunks)
  # FIX 5: added line breaks so background, question and answer do not stick together
  prompt = """You are an professional AI Assistant.
  Answer user question using ONLY document background below.
  Do not invent or assume facts.
  If information not available ,  simply say Information Missing.
  Document Background :\n""" + background + "\nUser Question : " + question + "\nAnswer:"
  client = Groq(api_key = key)
  response = client.chat.completions.create(
    model = model_name,
    messages = [ {"role" : "user",
                  "content" : prompt}],
                  temperature = 0.3,
  )
  return response.choices[0].message.content
