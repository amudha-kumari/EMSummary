#!/usr/bin/env python3
"""
retriever_setup.py

Set up a document retriever using Haystack. Supports both BM25 and dense retrievers
using SentenceTransformers. You can specify the retriever type and embedding model
via command-line arguments.

Example usage:
    python retriever_setup.py --retriever bm25
    python retriever_setup.py --retriever dense --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore


def setup_retriever(retriever_type: str, model_name: str = None):
    """
    Initializes a Haystack retriever based on the specified type.

    Args:
        retriever_type (str): Type of retriever to use ('bm25' or 'dense').
        model_name (str): Optional. Model name for dense retriever.

    Returns:
        retriever: An instance of BM25Retriever or EmbeddingRetriever.
    """
    doc_store = InMemoryDocumentStore()

    if retriever_type == "bm25":
        doc_store = InMemoryDocumentStore(use_bm25=True)
        retriever = BM25Retriever(document_store=doc_store)
        print("BM25 Retriever initialized.")
    elif retriever_type == "dense":
        if not model_name:
            raise ValueError("Model name must be provided for dense retriever.")
        retriever = EmbeddingRetriever(document_store=doc_store, embedding_model=model_name)
        doc_store.update_embeddings(retriever)
        print(f"Dense Retriever initialized with model: {model_name}")
    else:
        raise ValueError("Invalid retriever type. Choose 'bm25' or 'dense'.")

    return retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a document retriever using Haystack.")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["bm25", "dense"], help="Type of retriever to use.")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedding model name (required for dense retriever).")

    args = parser.parse_args()
    setup_retriever(args.retriever, args.model)

