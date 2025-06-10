import os
import psycopg2
import numpy as np
import fitz  # PyMuPDF
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# === External HDD paths ===
HDD_PATH = "/media/username/ExternalHDD/ai_vector/"
FAISS_INDEX_PATH = os.path.join(HDD_PATH, "faiss.index")
ID_MAP_PATH = os.path.join(HDD_PATH, "id_map.pkl")

# === PostgreSQL config ===
DB_CONFIG = {
    "dbname": "ragdb",
    "user": "raguser",
    "password": "ragpassword",
    "host": "localhost",
    "port": 5432
}

EMBED_DIM = 1024  # e5-large-v2
model = SentenceTransformer("intfloat/e5-large-v2")

# Connect DB
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

def chunk_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        for sent in text.split(". "):
            clean = sent.strip().replace("\n", " ")
            if clean:
                chunks.append({
                    "content": clean,
                    "page": page_num + 1,
                    "title": os.path.basename(pdf_path),
                    "chapter": "N/A"
                })
    return chunks

def insert_metadata(chunks):
    insert_query = """
    INSERT INTO chunks (content, page, title, chapter)
    VALUES (%s, %s, %s, %s)
    RETURNING id
    """
    ids = []
    for c in chunks:
        cur.execute(insert_query, (
            c["content"], c["page"], c["title"], c["chapter"]
        ))
        ids.append(cur.fetchone()[0])
    conn.commit()
    return ids

def build_faiss(chunks, ids):
    texts = [f"passage: {c['content']}" for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).astype('float32')
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ID_MAP_PATH, "wb") as f:
        pickle.dump(ids, f)

if __name__ == "__main__":
    PDF_PATH = "yourfile.pdf"
    chunks = chunk_pdf(PDF_PATH)
    print(f"[+] Extracted {len(chunks)} chunks")
    ids = insert_metadata(chunks)
    print("[+] Inserted into PostgreSQL")
    build_faiss(chunks, ids)
    print("[+] FAISS index saved to HDD")
    cur.close()
    conn.close()