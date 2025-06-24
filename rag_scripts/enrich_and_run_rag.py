# enrich_and_run_rag.py
# Author: Dr. Amudha Kumari Duraisamy

import argparse
import pandas as pd
import time
from Bio import Entrez
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.schema import Document
from transformers import pipeline


def fetch_pubmed_abstract(doi_or_title):
    search_term = f"{doi_or_title}[DOI]" if "/" in doi_or_title else doi_or_title
    try:
        handle = Entrez.esearch(db="pubmed", term=search_term)
        record = Entrez.read(handle)
        handle.close()
        if not record["IdList"]:
            return None

        pubmed_id = record["IdList"][0]
        time.sleep(0.4)  # Respect API rate limits
        handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="abstract", retmode="text")
        abstract = handle.read().strip()
        handle.close()
        return abstract
    except Exception as e:
        print(f"PubMed fetch error for '{doi_or_title}': {e}")
        return None


def enrich_metadata(row):
    base_text = ". ".join([row["Title"], row.get("Publication_Title", ""), row.get("Abstract", ""), row.get("Keywords", "")])
    doi = row.get("DOI", "")
    pubmed_text = fetch_pubmed_abstract(doi or row.get("Publication_Title", ""))
    enriched = f"{base_text}\n\nPubMed: {pubmed_text}" if pubmed_text else base_text
    return enriched.strip()


def setup_rag(metadata_list, summarizer_model, use_gpu=False):
    document_store = InMemoryDocumentStore(use_bm25=True)
    documents = [Document(content=txt, meta={"source": f"entry_{i}"}) for i, txt in enumerate(metadata_list)]
    document_store.write_documents(documents)

    retriever = BM25Retriever(document_store=document_store)
    device = 0 if use_gpu else -1
    summarizer = pipeline("summarization", model=summarizer_model, device=device)

    return retriever, summarizer


def run_rag(retriever, summarizer, query_text):
    docs = retriever.retrieve(query=query_text, top_k=5)
    full_text = " ".join([doc.content for doc in docs])
    result = summarizer(full_text, max_length=250, min_length=120, do_sample=False)
    return result[0]['summary_text']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", required=True, help="Path to EMDB metadata TSV")
    parser.add_argument("--model", required=True, help="HuggingFace summarizer model (e.g., facebook/bart-large-cnn)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--output_tsv", required=True, help="Where to save generated summaries")
    parser.add_argument("--num_entries", type=int, default=5)
    parser.add_argument("--entrez_email", type=str, required=True, help="your email for Entrez API usage")
    args = parser.parse_args()

    Entrez.email = args.entrez_email

    df = pd.read_csv(args.input_tsv, sep="\t").fillna("").head(args.num_entries)
    print("\n🔎 Enriching metadata with PubMed abstracts...")
    df["enriched_text"] = df.apply(enrich_metadata, axis=1)

    retriever, summarizer = setup_rag(df["enriched_text"].tolist(), args.model, args.gpu)

    print("\n📄 Generating RAG summaries...")
    df["RAG_Summary"] = df["enriched_text"].apply(lambda txt: run_rag(retriever, summarizer, txt))

    df[["EMDB_ID", "EM_Method", "RAG_Summary"]].to_csv(args.output_tsv, sep="\t", index=False)
    print(f"\n Summaries saved to {args.output_tsv}")


if __name__ == "__main__":
    main()

