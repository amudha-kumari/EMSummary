#!/usr/bin/env python3
"""
rag_main.py

A self-contained script that builds and runs a simple Retrieval-Augmented Generation (RAG) pipeline
using Haystack v1.x. Supports metadata input from file or via EMDB API.

Usage:
    python rag_main.py  --emdb_id 60072 --model facebook/bart-large-cnn
    python rag_main.py --id_file emdb_ids.txt --model ...
"""

import argparse
import subprocess
import os
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack.schema import Document
from transformers import pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_SCRIPT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "scripts", "extract_emdb_metadata_via_api.py"))

def fetch_metadata_via_api(emdb_id=None, id_file=None, output_file="temp_metadata.tsv"):
    cmd = ["python", EXTRACT_SCRIPT_PATH]
    if emdb_id:
        cmd.extend(["-i", emdb_id])
    elif id_file:
        cmd.extend(["-f", id_file])
    cmd.extend(["-o", output_file])
    subprocess.run(cmd, check=True)
    return ""

def load_metadata(file_path):
    """Load metadata text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def initialize_document_store(retriever_type):
    """Initialize the document store with or without BM25."""
    if retriever_type == "bm25":
        return InMemoryDocumentStore(use_bm25=True)
    else:
        return InMemoryDocumentStore()

def populate_documents(document_store):
    """Populate the document store with example documents. Replace with real content."""
    documents = [Document(content=document_store["content"], meta=document_store["meta"]) ]
    document_store.write_documents(documents)

def setup_retriever(document_store, retriever_type, use_gpu=False):
    """Set up BM25 or dense retriever."""
    if retriever_type == "bm25":
        return BM25Retriever(document_store=document_store)
    elif retriever_type == "dense":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=use_gpu
        )
        document_store.update_embeddings(retriever)
        return retriever
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")

def setup_summarizer(model_name: str, use_gpu: bool):
    """Set up a Hugging Face summarization pipeline directly."""
    device = 0 if use_gpu else -1
    return pipeline("summarization", model=model_name, device=device)

def run_rag_pipeline(retriever, summarizer, query: str, metadata: str = None):
    """Run retrieval + summarization using Hugging Face directly."""
    full_query = f"{metadata}\n\n{query}" if metadata else query
    docs = retriever.retrieve(query=full_query, top_k=5)
    full_text = " ".join([doc.content for doc in docs])
    result = summarizer(full_text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG-style Haystack pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Summarizer model name or path.")
    parser.add_argument("--metadata_file", type=str, help="Optional metadata file (e.g. title, abstract).")
    parser.add_argument("--emdb_id", type=str, help="Fetch metadata from EMDB API using a single ID.")
    parser.add_argument("--id_file", type=str, help="Fetch metadata from EMDB API using a file of IDs.")
    parser.add_argument("--retriever_type", type=str, default="bm25", choices=["bm25", "dense"],
                        help="Choose BM25 or dense retriever.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for dense retriever and summarizer.")
    parser.add_argument("--out_file", type=str, help="Write generated summary to file.")

    args = parser.parse_args()

    # Load metadata via file, EMDB ID or ID file (priority order)
    if args.metadata_file:
        metadata_text = load_metadata(args.metadata_file)
        query = "Generate a concise summary"
    elif args.emdb_id or args.id_file:
        metadata_text = fetch_metadata_via_api(emdb_id=args.emdb_id, id_file=args.id_file)
        # Build query automatically based on EMDB ID(s)
        if args.emdb_id:
            query = f"Generate a 150-word summary for EMD-{args.emdb_id}"
        else:
            # if id_file, summarize all ids (or make a generic query)
            query = "Generate a 150-word summary for provided EMDB entries"
    else:
        print("[ERROR] Please provide either --metadata_file or --emdb_id or --id_file")
        exit(1)

    document_store = initialize_document_store(args.retriever_type)
    populate_documents(document_store)
    retriever = setup_retriever(document_store, args.retriever_type, use_gpu=args.gpu)
    summarizer = setup_summarizer(args.model, use_gpu=args.gpu)

    result = run_rag_pipeline(retriever, summarizer, query, metadata_text)
    print("\n📢 Output:\n" + result)

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(result)
        print(f"Output written to {args.out_file}")

    # Clean up the temporary metadata file
    if os.path.exists("temp_metadata.tsv"):
        try:
            os.remove("temp_metadata.tsv")
            print(" Temporary file 'temp_metadata.tsv' removed.")
        except Exception as e:
            print(f"[ERROR] Failed to remove temporary file: {e}")
