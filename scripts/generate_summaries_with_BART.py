"""
generate_summaries_with_BART.py

This script is designed to generate summaries for EMDB entries using a Natural Language Processing (NLP) model.
It extracts relevant metadata (e.g., Title, Abstract, Keywords, Publication Title, Sample) from EMDB entry XML files,
and generates a concise summary for each entry.

The summary generation is based on an NLP model (such as TextRank or BART-lange-cnn or other summarization models), and the resulting summaries
are saved to a specified output file.

Usage:
    python generate_summaries_with_BART.py --in_file /path/to/metadata_file --out_file out_file.tsv

Arguments:
    --in_file     Path to the directory containing input data for summary generation.
    --out_file    Path to the output file where the generated summaries will be saved. The output will be in a tab-separated values (TSV) format.

Functions:
    - load_data: Loads the metadata from the provided XML files.
    - generate_summary: Generates a summary for each EMDB entry using an NLP model.
    - save_summaries: Saves the generated summaries to the specified output file.

Requirements:
    - Python 3.x
    - transformers (for NLP models)
    - pandas (for data handling)
    - any other necessary dependencies for NLP and XML processing

Note:
    The script expects the TSV file to have a specific structure, and each entry should contain relevant metadata
    (e.g., Title, Abstract, Keywords, Sample, etc.) that can be used to generate summaries.

Author:
    Dr. rer. nat. Amudha Kumari Duraisamy
email:
    amudhakumari@gmail.com
"""

import os
import argparse
import pandas as pd
from sumy.nlp.tokenizers import Tokenizer
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline
import nltk

# Add this path if NLTK data is stored in a non-standard location
nltk.data.path.append('/Users/amudha/nltk_data')


# Function to summarize using Sumy (TextRank)
def sumy_summary(text, method="TextRank"):
    # First, use nltk's word_tokenize to split the text into words
    word_tokens = word_tokenize(text)
    cleaned_text = ' '.join(word_tokens)  # Join back into a string for Sumy

    # Pass 'english' instead of sent_tokenize to Tokenizer
    parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))

    if method == "TextRank":
        summarizer = TextRankSummarizer()
    else:
        raise ValueError(f"Unsupported summarization method: {method}")

    summary_sentences = summarizer(parser.document, sentences_count=3)
    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return summary


# Function to summarize using BERT (Hugging Face pipeline)
def bert_summary(text):
    # Clean the text by removing excess spaces
    cleaned_text = ' '.join(text.split())

    # Load Hugging Face's BERT summarizer pipeline (BART model)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Perform summarization
    summary = summarizer(cleaned_text, max_length=150, min_length=50, do_sample=False)

    # Get the first summary result (as it's a list of dictionaries)
    return summary[0]['summary_text']


# Function to load and process metadata from the TSV file
def load_metadata(file_path):
    return pd.read_csv(file_path, sep='\t')


# Function to generate summaries from Title, Publication Title, and Abstract
def generate_summaries(metadata):
    summaries = []
    for index, row in metadata.iterrows():
        # Initialize a list to collect valid text parts
        text_parts = [row['Title']]

        # Add Publication_Title if it's valid
        if pd.notna(row['Publication_Title']) and row['Publication_Title'].strip().lower() != "unknown":
            text_parts.append(row['Publication_Title'])

        if pd.notna(row['Abstract']) and row['Abstract'].strip().lower() != "unknown":
            text_parts.append(row['Abstract'])

        if pd.notna(row['Keywords']) and row['Keywords'].strip().lower() != "unknown":
            text_parts.append(row['Keywords'])

        if pd.notna(row['PDB_ID']) and row['PDB_ID'].strip().lower() != "unknown":
            text_parts.append(row['PDB_ID'])

        text_parts.append(row['Sample_Names'])
        text_parts.append(row['Organisms'])
        text_parts.append(row['EM_Method'])
        text_parts.append(str(row['Resolution']))

        # Join the valid parts into the full_text string
        full_text = ". ".join(text_parts)

        # Clean and check if the text is long enough for BERT
        if len(full_text.split()) < 50:  # Arbitrary: if fewer than 50 words
            summary = sumy_summary(full_text, method="TextRank")
        else:
            summary = bert_summary(full_text)

        summaries.append({
            "EMDB_ID": row["EMDB_ID"],
            "EM_Method": row["EM_Method"],
            "NLP_Method": "BART-large-cnn" if len(full_text.split()) >= 50 else "TextRank",
            "Summary": summary
        })

    return pd.DataFrame(summaries)


# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate summaries for EMDB entries with BERT-large-CNN model.")
    parser.add_argument('--in_file', type=str, required=True, help="Path to the file having metadata for generating summaries.")
    parser.add_argument('--out_file', type=str, required=True,
                        help="Output file name for writing the generated summaries (TSV or CSV).")

    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print(f"file '{args.in_File}' not found.")
        return

    metadata = load_metadata(args.in_file)

    # Generate summaries
    summarized_data = generate_summaries(metadata)

    # output_file = os.path.join(output_folder, "EM_summaries.tsv")
    summarized_data.to_csv(args.out_file, sep='\t', index=False)
    print(f"Summarized metadata saved to: {args.out_file}")


if __name__ == "__main__":
    main()
