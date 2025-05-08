"""
finetune_bart_emdb.py

This script fine-tunes the Hugging Face `facebook/bart-large-cnn` model on EMDB metadata and corresponding generated summaries. 
It is intended for training a domain-adapted summarization model for structural biology datasets.

Usage:
    python finetune_bart_emdb.py --metadata path/to/metadata.tsv --summaries path/to/summaries.tsv --out_dir path/to/save_model

Arguments:
    --metadata   Path to the TSV file containing EMDB metadata. This file should include fields like Title, Abstract, Keywords, etc.
    --summaries  Path to the TSV file containing EMDB_ID, NLP_Method, and the generated summary.
    --out_dir    Directory where the fine-tuned model and tokenizer will be saved.

Dependencies:
    - pandas
    - transformers
    - datasets
    - scikit-learn
    - torch

Author:
    Dr. rer. nat. Amudha Kumari Duraisamy
email:
    amudhakumari@gmail.com
"""

import pandas as pd
import argparse
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os
import torch
from safetensors.torch import load_file


def prepare_dataset(metadata_path, summaries_path):
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    summaries_df = pd.read_csv(summaries_path, sep='\t')

    # Merge on EMDB_ID
    merged_df = pd.merge(summaries_df, metadata_df, on="EMDB_ID", how="inner")

    # Combine relevant fields
    merged_df["text"] = merged_df[["Title", "Publication_Title", "Abstract", "Sample", "Keywords"]].fillna("").agg(" ".join, axis=1)
    merged_df["summary"] = merged_df["Summary"]

    return Dataset.from_pandas(merged_df[["text", "summary"]])

def tokenize_function(examples, tokenizer, text_column="text", max_input_length=1024):
    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=max_input_length)

def tokenize_summary(examples, tokenizer, summary_column="summary", max_target_length=256):
    model_inputs = tokenizer(examples[summary_column], truncation=True, padding="max_length", max_length=max_target_length)
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

def converting_safetensors_to_pytorch_model_bin(out_dir):
    # Manual conversion from safetensors to pytorch_model.bin
    safetensor_path = os.path.join(out_dir, "model.safetensors")
    bin_path = os.path.join(out_dir, "pytorch_model.bin")

    state_dict = load_file(safetensor_path)
    torch.save(state_dict, bin_path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART on EMDB summaries")
    parser.add_argument("--metadata", required=True, help="Path to input metadata file (TSV)")
    parser.add_argument("--summaries", required=True, help="Path to generated summaries file (TSV)")
    parser.add_argument("--out_dir", required=True, help="Directory to save fine-tuned model")
    args = parser.parse_args()

    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # device = torch.device("cpu")
    # model = model.to(device)

    # Load and prepare dataset
    dataset = prepare_dataset(args.metadata, args.summaries)
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized = tokenized.map(lambda x: tokenize_summary(x, tokenizer), batched=True)

    train_test = tokenized.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=os.path.join(args.out_dir, 'logs')
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    converting_safetensors_to_pytorch_model_bin(args.out_dir)

if __name__ == "__main__":
    main()

