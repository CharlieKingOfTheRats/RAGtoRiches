# ingest.py

import os
import psycopg2
import numpy as np
from sqlalchemy import create_engine, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber
import docx

# CONFIGURATION
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
model = SentenceTransformer("intfloat/e5-large-v2")

# Connect to PostgreSQL
engine = create_engine(PG_CONN_STRING)

# Parse file depending on extension
def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = parse_pdf(file_path)
    elif ext == ".docx":
        text = parse_docx(file_path)
    else:
        raise ValueError("Unsupported file type.")
    return text

# PDF parser
def parse_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# DOCX parser
def parse_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Chunk text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

# Embed using e5-large-v2
def embed_text(chunk):
    embedding = model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

# Store in PostgreSQL
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

# MAIN INGESTION PIPELINE
def ingest_file(file_path):
    filename = os.path.basename(file_path)
    text = parse_file(file_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        store_embedding(filename, i, chunk, embedding)

if __name__ == "__main__":
    while True:
        file_path = input("Enter path to file (or type 'quit' to exit): ")
        if file_path.lower() == "quit":
            break
        try:
            ingest_file(file_path)
            print(f"✅ Ingestion complete for {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")