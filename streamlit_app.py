import streamlit as st
from rag import rag_query_streaming

# --- Page config ---
st.set_page_config(
    page_title="SEC 10-K RAG",
    page_icon="📄",
    layout="centered"
)

st.title("📄 SEC 10-K Financial Intelligence")
st.markdown("Ask questions about annual reports from major financial companies.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    company = st.selectbox(
        "Select Company",
        options=["JPMorgan Chase", "PayPal", "Visa"],
        index=0
    )
    st.markdown("---")
    st.caption("Data source: SEC 10-K filings")
    st.caption("Embeddings: llama-text-embed-v2")
    st.caption("LLM: Gemini 3.1 Flash Lite")

# --- Company name → Pinecone metadata filter map ---
# COMPANY_FILTER_MAP = {
#     "JPMorgan Chase": "JPMorgan Chase",
#     "PayPal": "PayPal",
#     "Visa": "Visa"
# }

# --- Main input ---
question = st.text_input(
    "Ask a question about the selected company's 10-K filing:",
    placeholder="e.g. What are the main sources of revenue?"
)

ask_button = st.button("Ask", type="primary")

# --- Query and display ---
if ask_button and question.strip():
    company_filter = company

    with st.spinner("Retrieving relevant chunks..."):
        stream, sources = rag_query_streaming(question, company_filter)

    st.markdown("### 📝 Answer")
    answer_placeholder = st.empty()
    full_answer = ""

    for chunk in stream:
        if chunk.text:
            full_answer += chunk.text
            answer_placeholder.markdown(full_answer)

    if sources:
        st.markdown("### 📚 Retrieved Chunks")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Chunk {i} — Similarity: {source['score']}"):
                st.markdown(f"**Company:** {source['company']}")
                st.markdown(f"**Text:**\n\n{source['text']}")

elif ask_button and not question.strip():
    st.warning("Please enter a question before clicking Ask.")