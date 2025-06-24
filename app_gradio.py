"""
EMDB Metadata Summarization Web App using Gradio and Haystack

This script provides a lightweight Gradio web interface for generating
summaries of EMDB (Electron Microscopy Data Bank) entries using a RAG-style
(Retrieval-Augmented Generation) pipeline built with Haystack and Hugging Face Transformers.

Key Features:
-------------
- Accepts an EMDB ID (e.g., 'EMD-1234') as user input.
- Loads corresponding metadata text from a local directory.
- Constructs a document store and retriever (BM25 or dense vector-based).
- Uses a pre-trained Hugging Face summarization model (e.g., BioBART) to generate a concise summary.
- Supports GPU acceleration for dense retrievers and summarizers.
- Designed for local experimentation and demo of summarization capabilities.

Modules and Technologies Used:
------------------------------
- Haystack (for document store and retriever setup)
- Hugging Face Transformers (for summarization pipeline)
- Gradio (for building the interactive UI)
- PyTorch or TensorFlow (used implicitly by the summarizer model)

Functions:
----------
- load_metadata: Loads the metadata file corresponding to an EMDB ID.
- build_pipeline: Sets up the document store, retriever, and summarizer.
- summarize_from_emdb: Runs retrieval + summarization pipeline and returns the summary.
- Gradio interface: Launches a web app with fields for EMDB ID, model name, retriever type, and GPU toggle.

How to Run:
-----------
1. Ensure all metadata files (e.g., 'EMD-1234.txt') are stored in a `metadata/` folder.
2. Install required dependencies: gradio, haystack, transformers, sentence-transformers.
3. Run the script: `python app_gradio.py`
4. Access the app at: http://localhost:7860

Note:
-----
- This tool is designed for research and prototype use and does not include persistent storage or async queuing.
- Metadata files must contain a header line followed by content.

Author:
Dr. Amudha Kumari Duraisamy
"""
import gradio as gr
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
        "python", "script/extract_emdb_metadata_via_api.py",
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


def launch_gradio():
    iface = gr.Interface(
        fn=run_pipeline,
        inputs=[
            gr.Textbox(label="EMDB ID"),
            gr.Textbox(label="Model Path or Hugging Face Model Name"),
            gr.Radio(["bm25", "dense"], label="Retriever Type", value="bm25"),
            gr.Checkbox(label="Use GPU", value=False),
        ],
        outputs=gr.Textbox(label="Generated Summary"),
        title="EMDB Entry Summarizer",
        description="Enter an EMDB ID and summarizer model. Metadata will be fetched automatically."
    )
    iface.launch()


if __name__ == "__main__":
    launch_gradio()
