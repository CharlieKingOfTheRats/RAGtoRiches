import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import tiktoken

# ────────────────────────────────────────────────
# CONFIG & LOGGING
# ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("❌ POSTGRES_CONNECTION_STRING is not set.")
engine = create_engine(PG_CONN_STRING)

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "endpoint")
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

embedder = SentenceTransformer("intfloat/e5-mistral-7b-instruct")  # 1048-dim
tokenizer = tiktoken.get_encoding("cl100k_base")

# ────────────────────────────────────────────────
# VECTOR SEARCH
# ────────────────────────────────────────────────
def search_similar_chunks(query, top_k=5, metric="cosine"):
    embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    operator = {
        "cosine": "<=>",
        "l2": "<->",
        "inner": "<#>"
    }.get(metric, "<=>")

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT doc_title, chunk_text, {embedding_str}::vector {operator} embedding AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT :top_k
        """), {"top_k": top_k}).fetchall()

    return rows, embedding

# ────────────────────────────────────────────────
# MODEL SELECTION LOGIC
# ────────────────────────────────────────────────
def select_model(token_count):
    if token_count < 800:
        return "gpt-3.5-turbo"
    elif token_count < 1800:
        return "gpt-4o-mini"
    else:
        return "gpt-4o"

# ────────────────────────────────────────────────
# GPT CALL
# ────────────────────────────────────────────────
def ask_openai(context, user_query):
    prompt = (
        "You are a System Safety engineer and an expert analyst. "
        "Use the provided context to answer the question precisely and concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\nAnswer:"
    )

    token_count = len(tokenizer.encode(prompt))
    logging.info(f"🔢 Prompt tokens: {token_count}")

    model_to_use = select_model(token_count)
    logging.info(f"🤖 Using model: {model_to_use}")

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )

    answer = response.choices[0].message.content
    return answer, token_count, model_to_use

# ────────────────────────────────────────────────
# STORE FEEDBACK / METADATA
# ────────────────────────────────────────────────
def store_feedback(query, answer, feedback, prompt_tokens, model_used):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO feedback (
                query, answer, user_feedback,
                query_time, prompt_tokens, model
            ) VALUES (
                :q, :a, :f, :qt, :pt, :m
            )
        """), {
            "q": query,
            "a": answer,
            "f": feedback,
            "qt": datetime.utcnow(),
            "pt": prompt_tokens,
            "m": model_used
        })

# ────────────────────────────────────────────────
# MAIN CLI LOOP
# ────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        user_query = input("\n🔎 Enter your question (or type 'quit'): ").strip()
        if user_query.lower() == "quit":
            break

        try:
            chunks, query_embedding = search_similar_chunks(user_query, metric="cosine")

            if not chunks:
                logging.warning("❌ No relevant context found.")
                print("⚠️ Sorry, no context found.")
                continue

            context = "\n".join([chunk[1] for chunk in chunks])
            for i, (title, text, distance) in enumerate(chunks):
                logging.info(f"📄 Match {i+1}: '{title}' | Distance: {distance:.4f}")

            answer, prompt_tokens, model_used = ask_openai(context, user_query)
            print(f"\n💡 Answer:\n{answer}")

            fb = input("\nWas this helpful? (yes/no/skip): ").strip().lower()
            if fb in ["yes", "no"]:
                store_feedback(user_query, answer, fb, prompt_tokens, model_used)

        except Exception as e:
            logging.error(f"❌ Error: {e}")
            print(f"❌ Error: {e}")
