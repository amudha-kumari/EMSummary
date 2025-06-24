#!/usr/bin/env python3
"""
rag_pipeline.py

Builds a simple Retrieval-Augmented Generation (RAG) pipeline in Haystack v1.x
using a retriever and a summarizer (generator). You provide a query and
optional metadata (e.g. title/abstract) to improve grounding.

Example usage:
    python rag_pipeline.py --query "Generate a 150-word summary for EMD-60072" --model facebook/bart-large-cnn --retriever_type bm25 --metadata_file meta.txt
"""

import argparse
from retriever_setup import setup_retriever
from generator_setup import setup_summarizer


def load_metadata(file_path):
    """Load metadata text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_rag_pipeline(retriever, summarizer, query: str, metadata: str = None):
    """
    Retrieve documents and generate a summary/answer using the summarizer.

    Args:
        retriever: Haystack retriever.
        summarizer: Haystack summarizer.
        query (str): Input query.
        metadata (str): Optional metadata to add to the query.

    Returns:
        str: Generated output.
    """
    combined_query = f"{metadata}\n\n{query}" if metadata else query
    retrieved_docs = retriever.retrieve(query=combined_query, top_k=5)
    generated = summarizer.predict(documents=retrieved_docs)
    return generated["answers"][0]["answer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple RAG-style pipeline using Haystack.")
    parser.add_argument("--query", type=str, required=True, help="Query to generate summary for.")
    parser.add_argument("--metadata_file", type=str, help="Optional metadata file (e.g. title + abstract).")
    parser.add_argument("--retriever_type", type=str, default="bm25", choices=["bm25", "dense"],
                        help="Retriever type to use.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for the summarizer.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")

    args = parser.parse_args()
    metadata = load_metadata(args.metadata_file) if args.metadata_file else None

    retriever = setup_retriever(args.retriever_type)
    summarizer = setup_summarizer(args.model, args.gpu)

    output = run_rag_pipeline(retriever, summarizer, args.query, metadata)
    print("\n📢 Output:\n" + output)
