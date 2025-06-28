# RAGtoRiches
Absolutely — here’s a **professional-grade `README.md`** for your MVP AI System Safety Assistant. It’s designed for GitHub **and** conference/demo sharing.

---

# 🛡️ PosAIdon: AI System Safety Assistant

> An AI-powered assistant for Systems & Safety Engineers — answer MIL-STD-882-based questions, retrieve technical documents, and reduce risk analysis time by 90%.

---

## 🚀 Demo Preview

Ask questions like:

```
"List hazard controls for loss of comms during underwater deployment."
"What is the difference between Severity Cat II and III in MIL-STD-882E?"
"Summarize the FMEA results from our propulsion system analysis."
```

And get concise, context-rich answers, powered by your own uploaded PDFs and standards documents.

---

## ⚙️ Tech Stack

| Layer      | Tools Used                                      |
| ---------- | ----------------------------------------------- |
| Backend    | Python, FastAPI, pgvector, SentenceTransformers |
| RAG Engine | Vector + full-text hybrid search (PostgreSQL)   |
| LLM Access | Azure OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5)     |
| Frontend   | React (Next.js) + Mantine UI                    |
| Embeddings | `intfloat/e5-mistral-7b-instruct` (1048-d)      |
| Hosting    | Azure (App Service / Container App) & Vercel    |

---

## 💡 Features

* ✅ **Ask any question** related to system safety, MIL-STD-882E, hazard analysis, or technical reports
* 📎 **Ingest your own PDFs** (.pdf, .docx) — deduplicated, token-aware, chunked
* 🔁 **Hybrid Search**: vector similarity + PostgreSQL full-text (BM25)
* 🔍 **Model Router**: Chooses GPT-3.5, 4o-mini, or GPT-4o based on complexity
* 💰 **Token Metering**: Tracks estimated OpenAI usage & cost
* 🧠 **Domain-Specific Prompting**: Tuned for engineers, not chat fluff
* 🗂 **Metadata Filtering**: Query only relevant documents
* 🔄 **Feedback Logging**: User feedback saved to DB for iteration

---

## 🛠️ Local Setup

### 🔧 Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Environment variables required:

```env
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/yourdb
AZURE_OPENAI_KEY=sk-...
```

### 💻 Frontend

```bash
cd frontend-next
npm install
npm run dev
```

Set `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📁 Project Structure

```
.
├── backend/
│   ├── ingest.py            ← Parses and stores embeddings
│   ├── query.py             ← Hybrid search, LLM call, reranker
│   └── main.py              ← FastAPI server with /ask endpoint
├── frontend-next/
│   └── pages/index.tsx      ← React UI with Mantine
└── README.md                ← You're here!
```

---

## 📦 Ingestion

Drop `.pdf` or `.docx` files into your ingestion folder and run:

```bash
python ingest.py
```

It will:

* Extract and chunk text (700 chars with 100 overlap)
* Normalize + embed using `e5-mistral-7b-instruct`
* De-duplicate by SHA-256
* Save to PostgreSQL with token counts

---

## 💬 API Usage

### POST `/ask`

```json
{
  "question": "What is the control strategy for a single-point failure?",
  "filter": "MIL-STD",
  "hybrid": true
}
```

Response:

```json
{
  "answer": "To control single-point failures, apply redundancy, monitoring, and ...",
  "model_used": "gpt-4o-mini",
  "tokens_used": 613,
  "estimated_cost": "$0.0019"
}
```

---

## 🧠 Target Use Cases

* 📜 MIL-STD-882E guidance retrieval
* 📉 FMEA/FMECA summarization
* 🚀 UUV & autonomous system hazard analysis
* 📊 Flight, ground, or naval safety system support
* ⚙️ Real-time engineering assistant for ISSS members

---

## 📣 Status & Roadmap

* [x] MVP Backend & API
* [x] RAG pipeline + model router
* [x] React + Next.js UI
* [ ] File upload via UI
* [ ] Role-based access control
* [ ] Azure & Vercel deployment
* [ ] Enterprise onboarding flow
