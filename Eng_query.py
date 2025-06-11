# query.py
# Copyright © 2025 PantheonAI. All rights reserved.

import os
import openai
import tiktoken
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy import create_engine, text

# Azure OpenAI Config
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2023-05-15"
AZURE_DEPLOYMENT_NAME_GPT4O = "gpt-4o"
AZURE_DEPLOYMENT_NAME_GPT35 = "gpt-35-turbo"

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION

# PostgreSQL connection string (ensure this env var is set)
PG_CONN_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
engine = create_engine(PG_CONN_STRING, pool_size=20, max_overflow=0)

# Initialize PostgreSQL logging table for queries
def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                user_goal TEXT,
                analysis_type TEXT,
                token_count INTEGER
            )
        """))

# Log query details in PostgreSQL
def log_query(user_goal, analysis_type, token_count):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO query_log (user_goal, analysis_type, token_count)
            VALUES (:user_goal, :analysis_type, :token_count)
        """), {
            'user_goal': user_goal,
            'analysis_type': analysis_type,
            'token_count': token_count
        })

# Estimate tokens for cost/control
def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Extract a short subject summary for routing and prompt tuning
def extract_subject(question, model=AZURE_DEPLOYMENT_NAME_GPT35):
    messages = [
        {"role": "system", "content": "Summarize this engineering or analysis request into 3-5 words."},
        {"role": "user", "content": f"'{question}'"}
    ]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0,
        max_tokens=30
    )
    return response.choices[0].message["content"].strip()

# Get website text content for context (can be replaced with a real technical knowledge base or docs)
def get_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"[ERROR] Failed to load content from {url}: {e}"

# Stub web search example for engineering topics; replace with your own sources or APIs
def web_search(query, max_results=3):
    print(f"[INFO] Running technical web search for query: {query}")
    return [
        "https://www.engineeringtoolbox.com/",
        "https://asmedigitalcollection.asme.org/",
        "https://www.sciencedirect.com/topics/engineering"
    ][:max_results]

# Main orchestration: generate an engineering or analyst expert response
def auto_tool_orchestrator(user_goal):
    print(f"\n[INFO] Starting engineering analysis generation for goal: {user_goal}")

    analysis_type = extract_subject(user_goal)
    print(f"[INFO] Extracted analysis type: {analysis_type}")

    keywords_trigger_search = ['analysis', 'report', 'simulation', 'model', 'design', 'failure', 'risk', 'assessment']
    if any(word in analysis_type.lower() for word in keywords_trigger_search):
        urls = web_search(user_goal)
        aggregated_text = ""
        for url in urls:
            text = get_website_text(url)
            if "[ERROR]" not in text:
                aggregated_text += text[:2000] + "\n\n"
    else:
        aggregated_text = ""

    model = AZURE_DEPLOYMENT_NAME_GPT4O if "complex" in analysis_type.lower() else AZURE_DEPLOYMENT_NAME_GPT35

    # Prompt tailored for engineering and analyst expert
    role_prompt = (
        "You are Enginuity, an expert engineering analyst and technical report generator.\n\n"
        "1. Tailor your output to the user's engineering or technical analysis request.\n"
        "2. Use precise, clear, and professional technical language.\n"
        "3. Format the output in JSON with relevant sections like:\n"
        "   - summary\n"
        "   - methodology\n"
        "   - results\n"
        "   - recommendations\n"
        "4. Provide actionable insights or risk assessments when relevant.\n"
        "User Request: " + user_goal + "\n\nContext:\n" + aggregated_text[:4000]
    )

    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user_goal}
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.3,
        max_tokens=700
    )

    output_text = response.choices[0].message["content"]
    token_count = estimate_tokens(output_text, model=model)

    # Log query with analysis type and token usage
    log_query(user_goal, analysis_type, token_count)

    return {
        "document": output_text,
        "analysis_type": analysis_type,
        "tokens_used": token_count
    }

if __name__ == "__main__":
    init_db()
    print("🤖 Enginuity – Engineering & Analyst Expert")
    print("Type 'quit' to exit.\n")

    while True:
        user_goal = input("Enter your engineering or analysis request: ")
        if user_goal.lower() == "quit":
            break

        result = auto_tool_orchestrator(user_goal)

        print("\n📄 Generated Report:\n")
        print(result["document"])
        print(f"\n🗂️ Analysis Type: {result['analysis_type']}")
        print(f"🔢 Tokens Used: {result['tokens_used']}")
        print("\n" + "-"*80 + "\n")