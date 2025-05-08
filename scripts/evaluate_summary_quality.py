"""
Script to compute ROUGE scores between generated summaries (predictions) and ground truth summaries (references),
including both per-entry and overall evaluation metrics.

The script performs the following:
- Reads two input text files: one with generated summaries (predictions) and another with reference summaries.
- Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores for each prediction-reference pair individually.
- Computes overall ROUGE scores across the entire dataset.
- Prints per-entry and overall scores to the console.
- Saves the scores to a TSV file, including:
    - Per-entry ROUGE scores
    - A blank line for separation
    - A row with overall scores labeled as "Overall"

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is commonly used to assess the quality of summarization and
measures the overlap between predicted and reference text based on n-grams and sequences.

Requirements:
- `evaluate` package (from Hugging Face) for ROUGE metric computation
- Input files must be UTF-8 encoded and contain one summary per line

Usage:
------
python evaluate_summary_quality.py --predictions predictions.txt --references references.txt --out_file rouge_scores.tsv


Author:
    Dr. rer. nat. Amudha Kumari Duraisamy
email:
    amudhakumari@gmail.com
"""

import argparse
import evaluate
import csv

def read_predictions_with_ids(file_path):
    """
    Reads a tab-separated file and returns a list of (ID, prediction) tuples.
    Assumes the format: EMDB_ID<TAB>Summary text
    """
    with open(file_path, "r", encoding="utf-8") as file:
        entries = []
        for line in file:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed line in predictions file: {line}")
            entries.append((parts[0], parts[1]))
        return entries

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]

def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE scores using EMDB IDs from predictions file.")
    parser.add_argument('--predictions', required=True, help="Path to predictions file (EMDB_ID<TAB>prediction per line)")
    parser.add_argument('--references', required=True, help="Path to references file (one per line)")
    parser.add_argument('--out_file', default="rouge_scores.tsv", help="Output TSV file path (default: rouge_scores.tsv)")
    args = parser.parse_args()

    prediction_entries = read_predictions_with_ids(args.predictions)
    references = read_file(args.references)

    if len(prediction_entries) != len(references):
        raise ValueError("Mismatch in number of predictions and references.")

    print("Calucating Per-Entry ROUGE Scores\n")
    rouge = evaluate.load("rouge")

    rows = []
    emdb_ids = []
    prediction_texts = []

    for (emdb_id, prediction), reference in zip(prediction_entries, references):
        emdb_ids.append(emdb_id)
        prediction_texts.append(prediction)

        result = rouge.compute(predictions=[prediction], references=[reference])

        rows.append({
            "EMDB_ID": emdb_id,
            "ROUGE-1": result["rouge1"],
            "ROUGE-2": result["rouge2"],
            "ROUGE-L": result["rougeL"],
            "ROUGE-Lsum": result["rougeLsum"]
        })

    # Compute overall score
    overall_result = rouge.compute(predictions=prediction_texts, references=references)
    print("Overall ROUGE Score:")
    for k, v in overall_result.items():
        print(f"  {k}: {v:.4f}")

    # Add overall to output
    rows.append({})
    rows.append({
        "EMDB_ID": "Overall",
        "ROUGE-1": overall_result["rouge1"],
        "ROUGE-2": overall_result["rouge2"],
        "ROUGE-L": overall_result["rougeL"],
        "ROUGE-Lsum": overall_result["rougeLsum"]
    })

    with open(args.out_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["EMDB_ID", "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nScores saved to: {args.out_file}")

if __name__ == "__main__":
    main()
