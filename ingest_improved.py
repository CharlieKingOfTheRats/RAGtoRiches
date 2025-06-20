import os
import hashlib
import pdfplumber
import docx
import tiktoken
import numpy as np
from sqlalchemy import create_engine, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# CONFIGURATION
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("❌ POSTGRES_CONNECTION_STRING is not set.")
engine = create_engine(PG_CONN_STRING)
model = SentenceTransformer("intfloat/e5-large-v2")  # 768-dim open-source embedding model
tokenizer = tiktoken.get_encoding("cl100k_base")

# File Parsing
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

# Chunking (tuned overlap + size)
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return splitter.split_text(text)

# Embedding
def embed_text(chunk):
    return model.encode(chunk, normalize_embeddings=True).tolist()

# Chunk Hashing
def compute_hash(chunk):
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()

# Token Counter
def count_tokens(chunk):
    return len(tokenizer.encode(chunk))

# Check if chunk already exists
def is_duplicate(chunk_hash):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 FROM documents WHERE chunk_hash = :h LIMIT 1"), {"h": chunk_hash})
        return result.scalar() is not None

# Store new chunk into DB
def store_embedding(filename, chunk_id, chunk, embedding, chunk_hash, token_count):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO documents (filename, chunk_id, chunk_text, embedding, chunk_hash, token_count)
            VALUES (:filename, :chunk_id, :chunk_text, :embedding, :chunk_hash, :token_count)
        """), {
            'filename': filename,
            'chunk_id': chunk_id,
            'chunk_text': chunk,
            'embedding': embedding,
            'chunk_hash': chunk_hash,
            'token_count': token_count
        })

# Batch folder ingestion
if __name__ == "__main__":
    folder_path = input("📂 Enter folder path of files to ingest (or type 'quit'): ").strip()
    if folder_path.lower() == "quit":
        exit()

    for file_name in os.listdir(folder_path):
        if not file_name.endswith((".pdf", ".docx")):
            continue
        file_path = os.path.join(folder_path, file_name)
        print(f"\n📄 Processing: {file_name}")
        try:
            text = parse_file(file_path)
            chunks = chunk_text(text)
            new_chunks = 0
            for i, chunk in enumerate(chunks):
                chunk_hash = compute_hash(chunk)
                if is_duplicate(chunk_hash):
                    continue
                embedding = embed_text(chunk)
                store_embedding(file_name, i, chunk, embedding, chunk_hash, count_tokens(chunk))
                new_chunks += 1
            print(f"✅ Stored {new_chunks} new chunks from {file_name}")
        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")