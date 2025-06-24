"""
generate_finetuned_summaries.py

This script generates summaries from EMDB metadata using a fine-tuned BART model.

It reads an input TSV file containing metadata fields (such as title, abstract, keywords, etc.),
constructs a combined text prompt for each entry, and uses a fine-tuned BART model to generate
summaries. The results are saved in an output TSV file containing the EMDB ID, method, and generated summary.

Typical usage example:
    python generate_finetuned_summaries.py \
        --model_dir ../data/models/bart_emdb_finetuned_train/ \
        --model BART \
        --input_tsv ../data/metadata_emsummary_train.tsv \
        --output_tsv ../data/summaries_finetuned_bart_train.tsv \
        --num_entries 20

Arguments:
    --model_dir: Path to the directory containing the fine-tuned BART model and tokenizer.
    --input_tsv: Path to the input TSV file containing EMDB metadata entries.
    --output_tsv: Path to the output TSV file where generated summaries will be saved.
    --num_entries: (Optional) Number of metadata entries to summarize. Defaults to 5.

Notes:
    - The BART model used should be compatible with the Hugging Face Transformers library.
    - Maximum input length to the model is truncated to 1024 tokens as per BART's architecture.
    - Summaries are generated using beam search for higher-quality results.

Dependencies:
    - pandas
    - torch
    - transformers

Author:
    Dr. Amudha Kumari Duraisamy
email:
    amudhakumari@gmail.com
"""

import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import torch

def load_model_BART(model_dir):
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def load_model_T5(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def truncate_to_max_tokens(text, tokenizer, max_tokens=1024):
    # Tokenize and truncate to max_tokens
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def generate_summary(text, tokenizer, model, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Check if the tokenized input is empty or too small
    if inputs['input_ids'].size(1) == 0:
        raise ValueError("Tokenized input is empty.")

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=200,  # Increase max tokens for longer summary
            min_length=100,  # Increase min tokens to encourage longer summaries
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Generate summaries using a fine-tuned BART or faln-T5 models.")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--model", required=True, help="Model type (BART or T5)")
    parser.add_argument("--input_tsv", required=True, help="TSV file containing EMDB metadata")
    parser.add_argument("--output_tsv", required=True, help="Output TSV file to store generated summaries")
    parser.add_argument("--num_entries", type=int, default=5, help="Number of entries to process (default: 5)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_tsv, sep="\t").fillna("")
    df = df.head(args.num_entries)

    # Load model and tokenizer
    if args.model == "BART":
        tokenizer, model = load_model_BART(args.model_dir)
    elif args.model == "T5":
        tokenizer, model = load_model_T5(args.model_dir)


    def build_text(row):
        text_parts = [row["Title"]]
        if row["Publication_Title"].strip().lower() != "unknown":
            text_parts.append(row["Publication_Title"])
        if row["Abstract"].strip().lower() != "unknown":
            text_parts.append(row["Abstract"])
        if row["Keywords"].strip().lower() != "unknown":
            text_parts.append(row["Keywords"])

        return ". ".join([part for part in text_parts if part and part.strip()])

    df["text"] = df.apply(build_text, axis=1)

    # Truncate to 1024 tokens using tokenizer
    df["text"] = df["text"].apply(lambda x: truncate_to_max_tokens(x, tokenizer))

    # Generate summaries
    print(f"Generating summaries for {len(df)} entries...")
    df["Generated_Summary"] = df["text"].apply(lambda x: generate_summary(x, tokenizer, model))

    # Save output
    df[["EMDB_ID", "EM_Method", "Generated_Summary"]].to_csv(args.output_tsv, sep="\t", index=False)
    print(f"Summaries saved to: {args.output_tsv}")

if __name__ == "__main__":
    main()
