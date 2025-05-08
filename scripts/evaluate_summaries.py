"""
evaluate_summaries.py

This script evaluates the quality of automatically generated summaries for EMDB entries by comparing them
against reference texts derived from original metadata (Title, Publication Title, Abstract, Keywords, Sample).

The evaluation uses two main metrics:
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures lexical overlap.
- BERTScore: Measures semantic similarity using contextual embeddings from BERT.

The script requires two input files:
1. A metadata file containing reference metadata for each EMDB entry.
2. A summary file containing the generated summaries and metadata identifiers.

It merges these files on EMDB ID, compiles the reference text, and evaluates the generated summaries.

Usage:
    python evaluate_summaries.py --metadata /path/to/metadata.tsv --summaries /path/to/summaries.tsv --out_file /path/to/evaluation_results.tsv

Arguments:
    --metadata     Path to the TSV file containing original metadata (e.g., Title, Abstract, etc.).
    --summaries    Path to the TSV file with generated summaries and EMDB identifiers.
    --out_file       Path where the evaluation result TSV will be saved.

Output:
    A TSV file with per-entry ROUGE and BERTScore metrics:
        - ROUGE-1, ROUGE-2, ROUGE-L
        - BERTScore Precision, Recall, F1

Dependencies:
    - pandas
    - evaluate (Hugging Face's evaluation module)
    - bert_score

Note:
    This script assumes that the summaries and metadata align by EMDB ID and that summary generation was performed
    using fields such as Title, Abstract, Keywords, Publication Title, and Sample text.

Author:
    Dr. rer. nat. Amudha Kumari Duraisamy
email:
    amudhakumari@gmail.com
"""

import os
import argparse
import pandas as pd
from bert_score import score
import evaluate

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Function to compute ROUGE
def compute_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references)

# Function to compute BERTScore
def compute_bertscore(predictions, references):
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

# Prepare reference from multiple metadata fields
def build_reference_text(row):
    parts = [
        row.get("Title", ""),
        row.get("Publication_Title", ""),
        row.get("Abstract", ""),
        row.get("Sample", ""),
        row.get("Keywords", "")
    ]
    return " ".join(str(part).strip() for part in parts if pd.notna(part) and str(part).strip().lower() != "unknown")

# Evaluate all summaries
def evaluate_summaries(merged_df):
    results = []

    for _, row in merged_df.iterrows():
        prediction = row["Summary"]
        reference = row["Full_Reference"]

        rouge_score = compute_rouge([prediction], [reference])
        bert_score = compute_bertscore([prediction], [reference])

        results.append({
            "EMDB_ID": row["EMDB_ID"],
            "NLP_Method": row["NLP_Method"],
            "ROUGE-1": rouge_score["rouge1"],
            "ROUGE-2": rouge_score["rouge2"],
            "ROUGE-L": rouge_score["rougeL"],
            "BERTScore-P": bert_score["precision"],
            "BERTScore-R": bert_score["recall"],
            "BERTScore-F1": bert_score["f1"]
        })

    return pd.DataFrame(results)

# Main script
def main():
    parser = argparse.ArgumentParser(description="Extract metadata from EMDB XML files.")
    parser.add_argument('--metadata', type=str, required=True,
                        help="Path to the file having metadata used for generating summaries.")
    parser.add_argument('--summaries', type=str, required=True,
                        help="Path to the file having genrated summaries.")
    parser.add_argument('--out_file', type=str, required=True,
                        help="Output file name for writing the evaluation results (TSV or CSV).")

    args = parser.parse_args()

    # Load metadata and summaries
    metadata_df = pd.read_csv(args.metadata, sep='\t')
    summary_df = pd.read_csv(args.summaries, sep='\t')

    # Build full reference text
    metadata_df["Full_Reference"] = metadata_df.apply(build_reference_text, axis=1)

    # Merge summaries with reference text
    merged_df = pd.merge(summary_df, metadata_df[["EMDB_ID", "Full_Reference"]], on="EMDB_ID", how="inner")

    # Filter out empty references
    merged_df = merged_df[merged_df["Full_Reference"].str.strip() != ""]

    # Evaluate summaries
    eval_results = evaluate_summaries(merged_df)

    # Save results
    eval_results.insert(0, "Entry_No", range(1, len(eval_results) + 1))
    eval_results.to_csv(args.out_file, sep='\t', index=False)

if __name__ == "__main__":
    main()