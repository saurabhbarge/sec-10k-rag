# 📄 SEC 10-K Financial Intelligence — RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for querying SEC 10-K annual filings from major financial companies. Built with a focus on accurate financial document retrieval and real-time streaming responses.

🔗 **Live Demo:** [Launch the App](https://your-render-url.onrender.com)

---

## 🏗️ Architecture

User Query
│
▼
Streamlit UI
│
▼
Query Embedding (llama-text-embed-v2)
│
▼
Pinecone Vector Search (filtered by company)
│
▼
Gemini 3.1 Flash Lite (streaming response)
│
▼
Answer + Source Chunks

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| Embeddings | llama-text-embed-v2 (Pinecone Inference) |
| Vector Store | Pinecone |
| LLM | Gemini 3.1 Flash Lite |
| PDF Parsing | pypdf |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Containerisation | Docker |
| Deployment | Render |

---

## 📁 Project Structure

sec-10k-rag/
├── streamlit_app.py      # Streamlit UI with streaming response
├── rag.py                # Core RAG logic — embedding, retrieval, generation
├── main.py               # FastAPI backend
├── ingest.py             # PDF ingestion pipeline
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service local development
└── requirements.txt      # Dependencies

---

## 📊 Data

10-K filings ingested for the following companies (fiscal year 2025):

- JPMorgan Chase
- PayPal
- Visa

---

## 🔍 How It Works

1. **Ingestion** — SEC 10-K PDFs are parsed, cleaned, chunked (500 characters, 50 overlap), and embedded using `llama-text-embed-v2`
2. **Storage** — Embeddings are upserted to Pinecone with company metadata for filtered retrieval
3. **Query** — User query is semantically matched against relevant chunks using cosine similarity
4. **Generation** — Retrieved chunks are passed to Gemini as context, which streams a concise financial analyst response
5. **Display** — Answer streams token by token in the UI alongside source chunks and similarity scores

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11+
- Docker
- Pinecone API key
- Gemini API key

### Setup
```bash
# Clone the repo
git clone https://github.com/saurabhbarge/sec-10k-rag.git
cd sec-10k-rag

# Create .env file
echo "PINECONE_API_KEY=your_key_here" >> .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# Run with Docker Compose
docker-compose up --build
```

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

### Run without Docker
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🔌 API

### Health Check

GET /health

### Query

POST /query
Content-Type: application/json

**Request:**
```json
{
  "question": "What was JPMorgan Chase's total net revenue in 2025?",
  "company": "JPMorgan Chase"
}
```

**Response:**
```json
{
  "question": "What was JPMorgan Chase's total net revenue in 2025?",
  "company": "JPMorgan Chase",
  "answer": "JPMorgan Chase reported total net revenue of $182,447 million...",
  "sources": [
    {
      "text": "...",
      "score": 0.621,
      "company": "JPMorgan Chase"
    }
  ]
}
```

---

## 🗺️ Roadmap

### Retrieval Improvements
- Re-ranking with Cohere Rerank — improve chunk relevance ordering post-retrieval
- Hybrid search — combine dense vector search with BM25 keyword search for better recall
- Query Rewriter Agent — improve retrieval accuracy for complex and ambiguous queries

### Evaluation
- RAGAS evaluation framework — measure faithfulness, answer relevancy, and context precision
- Hallucination guardrails — detect and flag answers not grounded in retrieved context

### Agentic
- Cross-company Comparison Agent
- LangGraph agentic orchestration

### Data
- Temporal reasoning across fiscal years
- Additional companies (Goldman Sachs, Block Inc.)

---

## 👤 Author

**Saurabh Barge**  
Data Scientist 
[LinkedIn](https://www.linkedin.com/in/saurabh-barge/) | [GitHub](https://github.com/saurabhbarge)