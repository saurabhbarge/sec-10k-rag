import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Initialise clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("financial-rag")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialise text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

def ask_gemini(query, retrieved_chunks, model="gemini-2.5-flash-lite"):
    context = "\n\n".join([chunk["metadata"]["text"] for chunk in retrieved_chunks])

    prompt = f"""You are a senior financial analyst with expertise in analysing SEC 10-K filings.
Your job is to answer questions accurately and concisely using only the provided context.

Guidelines:
- Answer directly and concisely based strictly on the context provided
- If the context contains relevant financial figures, always include them in your answer
- If the context is insufficient to answer the question, say exactly: "The provided filings do not contain enough information to answer this question."
- Do not speculate or use any knowledge outside of the provided context
- Where relevant, mention the company name and fiscal year in your answer

Context:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text

def rag_query_filtered(query, company=None, namespace="financial-docs", top_k=3):
    # Step 1: Embed the query
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )

    # Step 2: Build filter if company is specified
    filter = {"company": {"$eq": company}} if company else None

    # Step 3: Retrieve relevant chunks from Pinecone
    results = index.query(
        namespace=namespace,
        vector=query_embedding[0]["values"],
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )

    # Step 4: Generate answer with Gemini
    answer = ask_gemini(query, results["matches"])

    return answer