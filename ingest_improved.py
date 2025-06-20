import os
import re
import uuid
import hashlib
import logging
import pdfplumber
import docx
import numpy as np
import tiktoken
from datetime import datetime
from sqlalchemy import create_engine, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────
# CONFIGURATION & SETUP
# ────────────────────────────────────────────────
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("❌ POSTGRES_CONNECTION_STRING is not set.")
engine = create_engine(PG_CONN_STRING)

model = SentenceTransformer("intfloat/e5-large-v2")  # 768-dim
tokenizer = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────
def sanitize_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', base)

def extract_doc_title(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else "Untitled"

def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def count_tokens(text):
    return len(tokenizer.encode(text))

# ────────────────────────────────────────────────
# FILE PARSING
# ────────────────────────────────────────────────
def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type.")

# ────────────────────────────────────────────────
# CHUNKING AND EMBEDDING
# ────────────────────────────────────────────────
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return [chunk.strip() for chunk in splitter.split_text(text) if count_tokens(chunk) >= 10]

def embed_text(chunk):
    return model.encode(chunk, normalize_embeddings=True).tolist()

# ────────────────────────────────────────────────
# DB OPERATIONS
# ────────────────────────────────────────────────
def is_duplicate(chunk_hash):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 FROM documents WHERE chunk_hash = :h LIMIT 1"), {"h": chunk_hash})
        return result.scalar() is not None

def store_chunk(doc_id, title, filename, chunk_id, chunk, embedding, chunk_hash, token_count):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO documents (
                doc_id, doc_title, filename, chunk_id,
                chunk_text, embedding, chunk_hash,
                token_count, ingest_time
            ) VALUES (
                :doc_id, :doc_title, :filename, :chunk_id,
                :chunk_text, :embedding, :chunk_hash,
                :token_count, :ingest_time
            )
        """), {
            "doc_id": doc_id,
            "doc_title": title,
            "filename": filename,
            "chunk_id": chunk_id,
            "chunk_text": chunk,
            "embedding": embedding,
            "chunk_hash": chunk_hash,
            "token_count": token_count,
            "ingest_time": datetime.utcnow()
        })

# ────────────────────────────────────────────────
# INGEST FOLDER
# ────────────────────────────────────────────────
def ingest_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if not file_name.endswith((".pdf", ".docx")):
            continue

        file_path = os.path.join(folder_path, file_name)
        logging.info(f"📄 Processing: {file_name}")

        try:
            raw_text = parse_file(file_path)
            if not raw_text.strip():
                logging.warning(f"⚠️ No extractable text in {file_name}")
                continue

            doc_id = str(uuid.uuid4())
            safe_name = sanitize_filename(file_name)
            title = extract_doc_title(raw_text)
            chunks = chunk_text(raw_text)

            new_chunks = 0
            for i, chunk in enumerate(chunks):
                chunk_hash = compute_hash(chunk)
                if is_duplicate(chunk_hash):
                    continue
                embedding = embed_text(chunk)
                store_chunk(doc_id, title, safe_name, i, chunk, embedding, chunk_hash, count_tokens(chunk))
                new_chunks += 1

            logging.info(f"✅ Stored {new_chunks} new chunks from {file_name}")

        except Exception as e:
            logging.error(f"❌ Failed to process {file_name}: {e}")

# ────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────
if __name__ == "__main__":
    folder_path = input("📂 Enter folder path of files to ingest (or type 'quit'): ").strip()
    if folder_path.lower() == "quit":
        exit()
    ingest_folder(folder_path)