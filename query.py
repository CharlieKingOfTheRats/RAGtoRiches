# query.py

import os
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# PostgreSQL config
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

# SentenceTransformer model
model = SentenceTransformer("intfloat/e5-large-v2")

# Azure OpenAI config
AZURE_OPENAI_ENDPOINT = "https://posaidon.openai.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

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

# Call Azure OpenAI 4o Mini model with retrieved context
def call_azure_openai(user_query, context_chunks):
    context_text = "\n\n".join([chunk[2] for chunk in context_chunks])

    system_prompt = (
        "You are an expert engineering analyst. Use the provided business documents to answer questions accurately.\n"
        "If the answer is not contained in the documents, say 'Insufficient data'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Documents:\n{context_text}\n\nQuestion:\n{user_query}"}
    ]

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=1200
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or type 'quit' to exit): ")
        if user_query.lower() == "quit":
            break

        chunks = search(user_query)
        answer = call_azure_openai(user_query, chunks)

        print("\n💡 Answer:\n")
        print(answer)
        print("\n" + "-"*80 + "\n")