#!/usr/bin/env python3
"""
rag_pipeline_with_initiaization.py

A self-contained script that builds and runs a simple Retrieval-Augmented Generation (RAG) pipeline
using Haystack v1.x. It supports BM25 and dense retrievers, and a Transformers-based summarizer.

Usage:
    python rag_pipeline_with_initiaization.py  --model facebook/bart-large-cnn
    python rag_pipeline_with_initiaization.py  --metadata_file meta.txt --retriever_type bm25 --model google/flan-t5-base
"""

import argparse
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack.schema import Document
from transformers import pipeline


def load_metadata(file_path):
    """Load metadata text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f) #skip first line (header)
        return f.read().strip()


def initialize_document_store(retriever_type):
    """Initialize the document store with or without BM25."""
    if retriever_type == "bm25":
        return InMemoryDocumentStore(use_bm25=True)
    else:
        return InMemoryDocumentStore()


def populate_documents(document_store, metadata_text: str = None):
    """Populate the document store with metadata."""
    documents = [Document(content=metadata_text, meta={"source": "metadata_file"})]
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
    """Set up a Hugging Face summarization pipeline directly (bypasses Haystack bugs)."""
    device = 0 if use_gpu else -1
    return pipeline("summarization", model=model_name, device=device)


def run_rag_pipeline(retriever, summarizer, metadata: str):
    """Run retrieval + summarization using only metadata as input."""
    if not metadata:
        raise ValueError("Metadata is required when no query is provided.")

    docs = retriever.retrieve(query=metadata, top_k=5)
    full_text = " ".join([doc.content for doc in docs])
    result = summarizer(full_text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG-style Haystack pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Summarizer model name or path.")
    parser.add_argument("--metadata_file", type=str, help="Optional metadata (e.g. title, abstract).")
    parser.add_argument("--retriever_type", type=str, default="bm25", choices=["bm25", "dense"],
                        help="Choose BM25 or dense retriever.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for dense retriever and summarizer.")
    parser.add_argument("--out_file", type=str, help="write output to file")

    args = parser.parse_args()

    metadata_text = load_metadata(args.metadata_file) if args.metadata_file else None
    document_store = initialize_document_store(args.retriever_type)
    populate_documents(document_store, metadata_text=metadata_text)
    retriever = setup_retriever(document_store, args.retriever_type, use_gpu=args.gpu)
    summarizer = setup_summarizer(args.model, use_gpu=args.gpu)

    result = run_rag_pipeline(retriever, summarizer, metadata_text)
    print("\n📢 Output:\n" + result)

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(result)
        print(f"Output written to {args.out_file}")
