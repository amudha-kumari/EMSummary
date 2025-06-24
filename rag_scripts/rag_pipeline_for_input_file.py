#!/usr/bin/env python3
"""
rag_pipeline_for.py

A self-contained script that builds and runs a simple Retrieval-Augmented Generation (RAG) pipeline
using Haystack v1.x. Supports BM25 and dense retrievers, and a Hugging Face summarizer.
Processes metadata line-by-line from an input file, generating a summary for each.

Usage:
    python rag_pipeline_for_input_file.py --query "Summarize EMD-60072" --model facebook/bart-large-cnn --metadata_file meta.txt
"""

import argparse
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack.schema import Document
from transformers import pipeline


def load_metadata_lines(file_path):
    """Load metadata lines from file, skipping the header and empty lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Skip the first line (header), then strip and filter empty lines
    return [line.strip() for line in lines[1:] if line.strip()]



def initialize_document_store(retriever_type):
    """Initialize the document store with or without BM25."""
    if retriever_type == "bm25":
        return InMemoryDocumentStore(use_bm25=True)
    else:
        return InMemoryDocumentStore()


def populate_documents(document_store):
    """Populate the document store with example documents. Replace with your real content."""
    sample_docs = [
    ]
    documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in sample_docs]
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
    parser = argparse.ArgumentParser(description="Run a RAG-style Haystack pipeline with line-by-line metadata.")
    parser.add_argument("--query", type=str, required=True, help="Base query for the generator.")
    parser.add_argument("--model", type=str, required=True, help="Summarizer model name or path.")
    parser.add_argument("--metadata_file", type=str, help="Optional metadata file with one metadata entry per line.")
    parser.add_argument("--retriever_type", type=str, default="bm25", choices=["bm25", "dense"],
                        help="Choose BM25 or dense retriever.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for dense retriever and summarizer.")
    parser.add_argument("--out_file", type=str, help="Write all outputs to this file.")

    args = parser.parse_args()

    metadata_lines = load_metadata_lines(args.metadata_file) if args.metadata_file else [None]
    document_store = initialize_document_store(args.retriever_type)
    populate_documents(document_store)
    retriever = setup_retriever(document_store, args.retriever_type, use_gpu=args.gpu)
    summarizer = setup_summarizer(args.model, use_gpu=args.gpu)

    results = []
    for md in metadata_lines:
        summary = run_rag_pipeline(retriever, summarizer, args.query, md)
        results.append({"metadata": md, "summary": summary})

    # Print each result
    for i, res in enumerate(results, 1):
        print(f"\n📢 Result {i} | Metadata: {res['metadata']}\n{res['summary']}")

    # Write all to file if requested
    if args.out_file:
        with open(args.out_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(f"Metadata: {res['metadata']}\nSummary:\n{res['summary']}\n\n")
        print(f"\nAll output written to {args.out_file}")

