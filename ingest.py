# ingest.py
# Ingestion pipeline for SEC 10-K filings
# Loads PDFs, cleans text, chunks, embeds, and upserts to Pinecone

import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialise Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("financial-rag")

# Initialise text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)


def clean_text(text):
    """Remove excessive newlines and spaces from raw PDF text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    return text


def upsert_documents(index, documents, model="llama-text-embed-v2", namespace="financial-docs", batch_size=50):
    """Embed and upsert documents to Pinecone in batches."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc["text"] for doc in batch]

        embeddings = pc.inference.embed(
            model=model,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )

        vectors = []
        for doc, embedding in zip(batch, embeddings):
            vectors.append({
                "id": doc["id"],
                "values": embedding["values"],
                "metadata": doc["metadata"] | {"text": doc["text"]}
            })

        index.upsert(vectors, namespace)
        print(f"  Upserted batch {i // batch_size + 1} / {-(-len(documents) // batch_size)}")


def process_company(company_info):
    """Load, clean, chunk, and upsert a single company's 10-K filing."""
    print(f"\nProcessing {company_info['company']}...")

    # Step 1: Extract text from PDF
    reader = PdfReader(company_info["path"])
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"
    print(f"  Characters extracted: {len(raw_text)}")

    # Step 2: Clean
    cleaned_text = clean_text(raw_text)

    # Step 3: Chunk
    chunks = splitter.split_text(cleaned_text)
    print(f"  Chunks created: {len(chunks)}")

    # Step 4: Attach metadata
    company_slug = company_info["company"].lower().replace(" ", "_")
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{company_slug}_10k_chunk_{i}",
            "text": chunk,
            "metadata": {
                "company": company_info["company"],
                "filing_type": "10-K",
                "fiscal_year": company_info["fiscal_year"],
                "chunk_index": i
            }
        })

    # Step 5: Upsert to Pinecone
    upsert_documents(index, documents)
    print(f"  Done: {company_info['company']}")


# --- Companies to ingest ---
# Update paths to point to your local PDF files before running
companies = [
    {"path": "data/JPM_10K.pdf", "company": "JPMorgan Chase", "fiscal_year": "2025"},
    {"path": "data/Paypal_10K.pdf", "company": "PayPal", "fiscal_year": "2025"},
    {"path": "data/Visa_10K.pdf", "company": "Visa", "fiscal_year": "2025"},
]

if __name__ == "__main__":
    for company in companies:
        process_company(company)
    print("\nAll data ingested successfully.")