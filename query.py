# This code runs with proper dependences, Azure Key, and Postgresql config

import os
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# Azure OpenAI Setup
endpoint = "#Azure_Endpoint"
deployment = "gpt-4o-mini"
api_version = "2024-12-01-preview"
api_key = os.getenv("AZURE_OPENAI_KEY")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

# PostgreSQL
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING environment variable is not set.")
engine = create_engine(PG_CONN_STRING)
model = SentenceTransformer("intfloat/e5-base-v2")  # Make sure this matches your DB vector size

def search_similar_chunks(user_query, top_k=5):
    query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    with engine.connect() as conn:
        sql = text(f"""
            SELECT filename, chunk_text, embedding <=> '{embedding_str}'::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT :top_k
        """)
        results = conn.execute(sql, {'top_k': top_k}).fetchall()
        return results

def ask_openai(context, user_query):
    prompt = (
        f"You are an expert engineering and technical analyst.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        f"Answer:"
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=800,
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_query = input("Ask your question (or type 'quit'): ")
        if user_query.lower().strip() == 'quit':
            break

        try:
            results = search_similar_chunks(user_query)
            context = "\n".join([r[1] for r in results])
            answer = ask_openai(context, user_query)
            print(f"\n💡 Answer:\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\nPlease try again.")
