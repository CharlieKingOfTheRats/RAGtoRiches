# query.py

import os
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# PostgreSQL config
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

# Embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

# Connect to PostgreSQL
engine = create_engine(PG_CONN_STRING)

# Vector similarity search
def search(query, top_k=5):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    sql = text("""
        SELECT filename, chunk_id, chunk_text,
            (embedding <=> :query_embedding) AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        results = conn.execute(sql, {
            'query_embedding': query_embedding,
            'top_k': top_k
        }).fetchall()

    return results

if __name__ == "__main__":
    while True:
        user_query = input("Enter your search query (or type 'quit' to exit): ")
        if user_query.lower() == "quit":
            break
        results = search(user_query)
        for res in results:
            print(f"\nFile: {res[0]}, Chunk: {res[1]}, Distance: {res[3]:.4f}")
            print(res[2][:500])  # show partial chunk