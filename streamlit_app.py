# resume_rag.py
import os
import json
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from typing import List, Dict
import math
import hashlib

# ---------- Config ----------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small & fast
INDEX_PATH = "resume_index.faiss"
META_PATH = "resume_metadata.json"
CHUNK_SIZE = 400   # characters (approx). adjust if you prefer token-based chunking
CHUNK_OVERLAP = 120
TOP_K = 5
GPT4ALL_MODEL = "ggml-gpt4all-j-v1.3-groovy.bin"  # example; gpt4all will download if allow_download=True

# ---------- Utilities ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    pieces = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + size
        chunk = text[start:end]
        pieces.append(chunk.strip())
        start = max(end - overlap, end)  # move with overlap
    return pieces

def hashed_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ---------- Index build/load ----------
def build_index(emb_model: SentenceTransformer, resume_text: str):
    # chunk
    chunks = chunk_text(resume_text)
    embeddings = emb_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # simple index; fast for small datasets
    index.add(embeddings)

    # metadata
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": i,
            "text": chunk,
            "hash": hashed_id(chunk),
        })

    # persist
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return index, metadata

def load_index_and_meta(emb_model: SentenceTransformer):
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, None
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    dim = emb_model.get_sentence_embedding_dimension()
    index = faiss.read_index(INDEX_PATH)
    return index, meta

# ---------- Retrieval ----------
def retrieve(emb_model: SentenceTransformer, index, metadata, query: str, top_k: int = TOP_K):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append({
            "score": float(D[0][i]),
            "text": metadata[idx]["text"],
            "id": metadata[idx]["id"]
        })
    return results

# ---------- LLM Answering (gpt4all) ----------
def answer_with_gpt4all(model_name: str, system_prompt: str, user_prompt: str, max_tokens=512):
    # model auto-downloads to cache dir if not present
    model = GPT4All(model_name, allow_download=True)
    with model.chat_session() as session:
        # Provide a short system role/instruction
        session.system(system_prompt)
        session.user(user_prompt)
        response = session.generate(max_tokens=max_tokens)
    return response

# ---------- Prompt builder ----------
def build_prompt(question: str, retrieved_chunks: List[Dict]):
    # Safety: instruct the model to ONLY use the provided context.
    context_texts = "\n\n---\n\n".join([f"[chunk {r['id']}]\n{r['text']}" for r in retrieved_chunks])
    system_prompt = (
        "You are an assistant that answers questions using ONLY the provided resume context. "
        "If the answer is not contained in the context, say: 'I don’t know based on the resume.' "
        "Be concise and cite the chunk id when referencing specifics."
    )
    user_prompt = (
        f"Context:\n{context_texts}\n\nQuestion: {question}\n\n"
        "Answer the question using only the context. If you cannot answer, say you don't know."
    )
    return system_prompt, user_prompt

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Resume RAG Chat", layout="centered")
    st.title("RAG Chatbot — Resume Assistant (local & free)")

    # Sidebar: upload resume
    st.sidebar.header("Resume (txt)")
    uploaded = st.sidebar.file_uploader("Upload a plain text resume (.txt). For PDF/DOCX see README below.", type=["txt"])
    if uploaded:
        resume_text = uploaded.read().decode("utf-8")
        st.sidebar.success("Uploaded resume.txt")
    else:
        if os.path.exists("resume.txt"):
            resume_text = open("resume.txt", "r", encoding="utf-8").read()
        else:
            resume_text = None

    # Load embedding model
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer(EMBED_MODEL_NAME)
    emb_model = load_embedding_model()

    # Build / load index
    index, meta = load_index_and_meta(emb_model)
    if index is None:
        if not resume_text:
            st.info("Upload resume.txt (sidebar) or place resume.txt next to this script, then click 'Build Index'.")
        if st.sidebar.button("Build index from uploaded resume"):
            if not resume_text:
                st.sidebar.error("No resume text found.")
            else:
                with st.spinner("Chunking + embedding + building FAISS index..."):
                    index, meta = build_index(emb_model, resume_text)
                st.sidebar.success("Index built & saved (resume_index.faiss, resume_metadata.json)")
    else:
        st.sidebar.success("Index loaded from disk.")

    st.write("### Chat")
    question = st.text_input("Ask a question about the resume", key="q")
    if st.button("Ask") and question.strip():
        if index is None or meta is None:
            st.error("Index not built. Upload resume and click Build index.")
        else:
            with st.spinner("Retrieving relevant resume passages..."):
                retrieved = retrieve(emb_model, index, meta, question, top_k=TOP_K)
            st.markdown("**Retrieved passages (source chunks):**")
            for r in retrieved:
                st.markdown(f"- chunk **{r['id']}** (score: {r['score']:.4f}): {r['text'][:300]}...")
            # Build prompt & answer
            system_prompt, user_prompt = build_prompt(question, retrieved)
            st.markdown("---")
            st.markdown("**Answer (from local LLM):**")
            with st.spinner("Generating answer with local model... (may take a few seconds)"):
                response = answer_with_gpt4all(GPT4ALL_MODEL, system_prompt, user_prompt, max_tokens=400)
            st.write(response)

    # Footer: quick instructions
    st.markdown("---")
    st.markdown("**Notes:** 1) For best results upload a plain text resume. 2) To re-index new resume delete `resume_index.faiss` and `resume_metadata.json` and rebuild.")
    st.markdown("**Advanced:** adapt chunking to tokens if you have a tokenizer; use a more advanced index (IVF/PQ) for very large docs.")

if __name__ == "__main__":
    main()
