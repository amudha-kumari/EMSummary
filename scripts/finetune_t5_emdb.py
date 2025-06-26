
"""
Fine-tune a T5 model for EMDB metadata summarization.

This script merges metadata and generated summaries, tokenizes the combined text,
and fine-tunes a T5 model (e.g., t5-base) for abstractive summarization using Hugging Face Transformers and Datasets.

Usage:
    python finetune_t5_emdb.py --metadata metadata.tsv --summaries summaries.tsv --out_dir output_directory --model_name t5-base --epochs 4 --batch_size 8 --eval_batch_size 8

Author: Dr. Amudha Kumari Duraisamy
"""

import pandas as pd
import os
import torch
import argparse
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import mlflow
import mlflow.pytorch



def prepare_dataset(metadata_path: str, summaries_path: str) -> Dataset:
    """
    Load and merge metadata and summary TSV files into a Hugging Face Dataset object.

    Args:
        metadata_path (str): Path to metadata TSV file.
        summaries_path (str): Path to summaries TSV file.

    Returns:
        Dataset: Hugging Face Dataset with 'text' and 'summary' columns.
    """
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    summaries_df = pd.read_csv(summaries_path, sep='\t')

    # Drop potential column name clashes before merging
    summaries_df = summaries_df.drop(columns=["EM_Method"], errors="ignore")

    # Merge datasets on EMDB ID
    merged_df = pd.merge(summaries_df, metadata_df, on="EMDB_ID", how="inner")

    # Concatenate selected metadata fields into a single 'text' field
    merged_df["text"] = merged_df[
        ["Title", "Publication_Title", "Abstract", "Sample_Names", "Keywords",
         "EM_Method", "Organisms", "Resolution", "PDB_ID"]
    ].fillna("").agg(" ".join, axis=1)

    # Rename summary column
    merged_df["summary"] = merged_df["Summary"]

    return Dataset.from_pandas(merged_df[["text", "summary"]])


def tokenize_function(examples, tokenizer, max_input_length=512, max_target_length=150):
    """
    Tokenize the input text and target summaries for T5 training.

    Args:
        examples (dict): A dictionary with 'text' and 'summary' fields.
        tokenizer: The T5 tokenizer.
        max_input_length (int): Maximum length of input tokens.
        max_target_length (int): Maximum length of target summary tokens.

    Returns:
        dict: Dictionary of tokenized inputs and labels.
    """
    # Prefix input with "summarize:" to guide T5 task
    inputs = ["summarize: " + text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    """
    Main function that handles argument parsing, dataset preparation, model training, and saving.
    """
    parser = argparse.ArgumentParser(description="Fine-tune T5 on EMDB metadata and summaries")
    parser.add_argument("--metadata", required=True, help="Path to metadata TSV file")
    parser.add_argument("--summaries", required=True, help="Path to summary TSV file")
    parser.add_argument("--out_dir", required=True, help="Output directory to save model")
    parser.add_argument("--model_name", default="t5-base", help="Pretrained model name (default: t5-base)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size per device")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare dataset
    dataset = prepare_dataset(args.metadata, args.summaries)
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    train_test = tokenized.train_test_split(test_size=0.1)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=os.path.join(args.out_dir, 'logs'),
        logging_steps=10,
        report_to="none"
    )

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    # Start MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("train_batch_size", args.batch_size)
        mlflow.log_param("eval_batch_size", args.eval_batch_size)

        # Train model
        trainer.train()

        # Evaluate and log metrics
        metrics = trainer.evaluate()
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model and artifacts
        model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts(args.out_dir, artifact_path="model_artifacts")

        print(f"Model and tokenizer saved to {args.out_dir}")


if __name__ == "__main__":
    main()
