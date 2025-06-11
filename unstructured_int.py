import os
import sys
import psycopg2
import numpy as np
from sqlalchemy import create_engine, text
from unstructured.partition.auto import partition  # unstructured.io OSS parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ================== CONFIGURATION ==================
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    print("ERROR: Please set the POSTGRES_CONNECTION_STRING environment variable.")
    sys.exit(1)

model = SentenceTransformer("intfloat/e5-large-v2")
engine = create_engine(PG_CONN_STRING)

def parse_file(file_path):
    print(f"Parsing file with unstructured.io: {file_path}")
    elements = partition(filename=file_path)  # unstructured usage here
    text = "\n".join([str(el) for el in elements])
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks")
    return chunks

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
    print(f"Stored chunk {chunk_id} in DB")

def ingest_file(file_path):
    filename = os.path.basename(file_path)
    text = parse_file(file_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        store_embedding(filename, i, chunk, embedding)

if __name__ == "__main__":
    while True:
        input_path = input("Enter the full path to the file to ingest (or type 'quit' to exit): ").strip()
        if input_path.lower() == "quit":
            print("Exiting program.")
            sys.exit(0)
        if not os.path.isfile(input_path):
            print(f"ERROR: File not found: {input_path}")
            continue
        ingest_file(input_path)
        print("Ingestion completed successfully!\n")