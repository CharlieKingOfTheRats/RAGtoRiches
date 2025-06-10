import os
import faiss
import pickle
import numpy as np
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# === Load environment variables ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

EMBED_DIM = 1024
model = SentenceTransformer("intfloat/e5-large-v2")

def retrieve_chunks_faiss(query, top_k=5):
    # Embed query
    query_vec = model.encode(f"query: {query}", normalize_embeddings=True).astype("float32").reshape(1, -1)

    # Load FAISS index and ID map
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(ID_MAP_PATH, "rb") as f:
        id_map = pickle.load(f)

    # Search
    D, I = index.search(query_vec, top_k)
    matched_ids = [id_map[i] for i in I[0]]

    # Retrieve metadata from PostgreSQL
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT content, page FROM chunks WHERE id = ANY(%s);", (matched_ids,)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# === CLI Interface ===
if __name__ == "__main__":
    while True:
        query = input("🔍 Ask a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        results = retrieve_chunks_faiss(query)
        print("\nTop context results:")
        for r in results:
            print(f"📄 Page {r[1]}: {r[0][:200]}...\n")

        # Format context
        context_text = "\n".join([f"(Page {r[1]}): {r[0]}" for r in results])

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer questions."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ],
            temperature=0.2
        )

        print("🤖 Answer:", response.choices[0].message["content"], "\n")