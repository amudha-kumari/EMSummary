import os
import argparse
import pandas as pd
from transformers import pipeline
import nltk
import torch

nltk.data.path.append('/Users/amudha/nltk_data')

# Function to summarize using BioBART
def biobart_summary(text, summarizer):
    cleaned_text = ' '.join(text.split())

    summary = summarizer(
        cleaned_text,
        max_length=150,
        min_length=50,
        do_sample=False,
        no_repeat_ngram_size=2
    )

    return summary[0]['summary_text']


# Load metadata
def load_metadata(file_path):
    return pd.read_csv(file_path, sep='\t')


# Generate summaries
def generate_summaries(metadata, summarizer):
    summaries = []
    for index, row in metadata.iterrows():
        text_parts = [row['Title']]

        if pd.notna(row['Publication_Title']) and row['Publication_Title'].strip().lower() != "unknown":
            text_parts.append(row['Publication_Title'])

        if pd.notna(row['Abstract']) and row['Abstract'].strip().lower() != "unknown":
            text_parts.append(row['Abstract'])

        if pd.notna(row['Keywords']) and row['Keywords'].strip().lower() != "unknown":
            text_parts.append(row['Keywords'])

        text_parts.append(str(row['Organisms']))
        text_parts.append(str(row['Resolution']))

        full_text = ". ".join(text_parts)

        summary = biobart_summary(full_text, summarizer)
        method_used = "biobart-v2-base"

        summaries.append({
            "EMDB_ID": row["EMDB_ID"],
            "EM_Method": row["EM_Method"],
            "NLP_Method": method_used,
            "Summary": summary
        })

    return pd.DataFrame(summaries)


# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Generate summaries for EMDB entries using BioBART.")
    parser.add_argument('--in_file', type=str, required=True, help="Path to input metadata file.")
    parser.add_argument('--out_file', type=str, required=True, help="Path to output summary file (TSV).")

    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print(f"File '{args.in_file}' not found.")
        return

    device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline(
        "summarization",
        model="GanjinZero/biobart-v2-base",
        tokenizer="GanjinZero/biobart-v2-base",
        device=device
    )

    metadata = load_metadata(args.in_file)
    summarized_data = generate_summaries(metadata, summarizer)
    summarized_data.to_csv(args.out_file, sep='\t', index=False)
    print(f"Summarized metadata saved to: {args.out_file}")


if __name__ == "__main__":
    main()
