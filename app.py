"""
app.py

Streamlit-based web application for summarizing EMDB (Electron Microscopy Data Bank) entry metadata
using NLP models integrated with Haystack's retrieval and summarization pipeline.

The app performs the following steps:
1. Accepts an EMDB ID, a Hugging Face summarization model path, retriever type (BM25 or dense), and GPU option.
2. Runs an external script to fetch metadata via EMDB's API and saves it in TSV format.
3. Loads the metadata into a Haystack `DocumentStore`.
4. Retrieves relevant segments using BM25 or dense embeddings with optional GPU support.
5. Summarizes the retrieved content using a transformer-based summarization pipeline (e.g., T5-small-finetuned-pubmed, BioBART).
6. Displays the generated summary in the web UI.

Dependencies:
- streamlit
- haystack
- transformers
- sentence-transformers
- pandas
- argparse (for the metadata script)
- subprocess
- csv

To run:
    streamlit run app.py

Author: Dr. Amudha Kumari Duraisamy
Project: EMSummary - Automated Metadata Summarization for EMDB Entries
"""

import streamlit as st
import subprocess
import os
import csv

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack import Document
from transformers import pipeline


def load_metadata(emdb_id):
    """Run extract_emdb_metadata_via_api.py and return metadata string for summarization."""
    output_path = f"data/{emdb_id}.tsv"
    os.makedirs("data", exist_ok=True)

    subprocess.run([
        "python", "scripts/extract_emdb_metadata_via_api.py",
        "-i", emdb_id,
        "-o", output_path
    ], check=True)

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        row = next(reader)
        return ". ".join(f"{k.replace('_', ' ')}: {v}" for k, v in row.items() if v)


def run_pipeline(emdb_id, model_path, retriever_type, use_gpu):
    """Run full metadata summarization pipeline."""
    # Step 1: Metadata
    metadata_text = load_metadata(emdb_id)

    # Step 2: Document store and retriever
    store = InMemoryDocumentStore(use_bm25=(retriever_type == "bm25"))
    store.write_documents([Document(content=metadata_text, meta={"emdb_id": emdb_id})])

    if retriever_type == "bm25":
        retriever = BM25Retriever(document_store=store)
    else:
        retriever = EmbeddingRetriever(
            document_store=store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=use_gpu
        )
        store.update_embeddings(retriever)

    # Step 3: Summarizer
    summarizer = pipeline("summarization", model=model_path, device=0 if use_gpu else -1)

    docs = retriever.retrieve(query=metadata_text, top_k=5)
    combined = " ".join(doc.content for doc in docs)
    summary = summarizer(combined, max_length=150, min_length=30, do_sample=False)

    return summary[0]["summary_text"]


# ---- Streamlit UI ----
st.set_page_config(page_title="EMDB Entry Summarizer", layout="centered")
st.title("EMDB Entry Summarizer")
st.markdown("Enter an EMDB ID and summarizer model. Metadata will be fetched automatically.")

with st.form("summarization_form"):
    emdb_id = st.text_input("EMDB ID", value="EMD-12345")
    model_path = st.text_input("Model Path or Hugging Face Model Name", value="amudhakumari/emdb-summary-generator")
    retriever_type = st.radio("Retriever Type", options=["bm25", "dense"], index=0)
    use_gpu = st.checkbox("Use GPU", value=False)

    submitted = st.form_submit_button("Generate Summary")

if submitted:
    try:
        with st.spinner("Running summarization pipeline..."):
            summary = run_pipeline(emdb_id, model_path, retriever_type, use_gpu)
        st.success("Summary generated successfully!")
        st.text_area("Generated Summary", value=summary, height=200)
    except Exception as e:
        st.error(f"Error: {e}")
