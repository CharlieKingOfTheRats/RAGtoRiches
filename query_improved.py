import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# PostgreSQL
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("❌ POSTGRES_CONNECTION_STRING is not set.")
engine = create_engine(PG_CONN_STRING)

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint="https://posaidon.openai.azure.com/"
)

model = SentenceTransformer("intfloat/e5-large-v2")  # Must match vector size in DB

# Search DB for relevant chunks
def search_similar_chunks(query, top_k=5, metric="cosine"):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    operator = {
        "cosine": "<=>",
        "l2": "<->",
        "inner": "<#>"
    }.get(metric, "<=>")

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT filename, chunk_text, embedding {operator} '{embedding_str}'::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT :top_k
        """), {"top_k": top_k}).fetchall()
        return rows

# Call Azure OpenAI
def ask_openai(context, user_query):
    prompt = (
        "You are an engineering and systems analyst. "
        "Use the provided context to answer the question precisely and concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\nAnswer:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )
    return response.choices[0].message.content

# Save optional user feedback
def store_feedback(query, answer, feedback):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO feedback (query, answer, user_feedback)
            VALUES (:q, :a, :f)
        """), {"q": query, "a": answer, "f": feedback})

# CLI loop
if __name__ == "__main__":
    while True:
        query = input("\n🔎 Enter your question (or type 'quit'): ").strip()
        if query.lower() == "quit":
            break
        try:
            rows = search_similar_chunks(query, metric="cosine")
            context = "\n".join([r[1] for r in rows])
            response = ask_openai(context, query)
            print(f"\n💡 Answer:\n{response}")

            fb = input("\nWas this helpful? (yes/no/skip): ").strip().lower()
            if fb in ["yes", "no"]:
                store_feedback(query, response, fb)
        except Exception as e:
            print(f"❌ Error: {e}")