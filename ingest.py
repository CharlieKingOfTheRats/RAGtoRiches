#This code runs for .pdf and .docx files with proper dependences, Azure key, and postgresql configuration

import os
import pdfplumber
import docx
import numpy as np
from sqlalchemy import create_engine, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# CONFIGURATION
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING environment variable is not set.")

engine = create_engine(PG_CONN_STRING)
model = SentenceTransformer("intfloat/e5-base-v2")  # 768 dimensions

def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower().strip()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            file_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == ".docx":
        doc = docx.Document(file_path)
        file_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type.")
    return file_text

def chunk_text(file_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(file_text)

def embed_text(chunk):
    embedding = model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

def store_embedding(filename, chunk_id, chunk, embedding):
    with engine.connect() as conn:
        sql = text("""
            INSERT INTO documents (filename, chunk_id, chunk_text, embedding)
            VALUES (:filename, :chunk_id, :chunk_text, :embedding)
        """)
        conn.execute(sql, {
            'filename': filename,
            'chunk_id': chunk_id,
            'chunk_text': chunk,
            'embedding': embedding
        })

if __name__ == "__main__":
    while True:
        file_path = input("Enter file path (or type 'quit'): ")
        if file_path.lower().strip() == 'quit':
            break
        try:
            file_text = parse_file(file_path)
            chunks = chunk_text(file_text)
            for i, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                store_embedding(os.path.basename(file_path), i, chunk, embedding)
            print(f"✅ Ingested {len(chunks)} chunks from {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
# This script ingests text files, splits them into chunks, embeds the chunks, and stores them in a PostgreSQL database.
