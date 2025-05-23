#!/usr/bin/env python3
"""
summarizer_setup.py

Initializes a summarization model using Haystack's TransformersSummarizer.
Works with Hugging Face models like 'facebook/bart-large-cnn', 'google/flan-t5-base', etc.

Example usage:
    python summarizer_setup.py --model facebook/bart-large-cnn
    python summarizer_setup.py --model google/flan-t5-base --gpu
"""

import argparse
from haystack.nodes import TransformersSummarizer


def setup_summarizer(model_name: str, use_gpu: bool = False):
    """
    Initializes a TransformersSummarizer with the specified model.

    Args:
        model_name (str): Hugging Face model path or name.
        use_gpu (bool): Use GPU if available.

    Returns:
        TransformersSummarizer: The summarization node.
    """
    summarizer = TransformersSummarizer(model_name_or_path=model_name, use_gpu=use_gpu)
    print(f"Summarizer initialized with model: {model_name} (GPU: {use_gpu})")
    return summarizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a summarizer using Haystack.")
    parser.add_argument("--model", type=str, required=True,
                        help="Name or path of the Hugging Face summarization model.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available (default: False).")

    args = parser.parse_args()
    setup_summarizer(args.model, args.gpu)
