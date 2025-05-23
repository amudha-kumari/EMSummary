import argparse
import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document
import json

def main(input_file, output_file=None):
    doc_store = InMemoryDocumentStore()
    documents = []

    df = pd.read_csv(input_file, sep='\t')

    for _, row in df.iterrows():
        content = f"Title: {row['Title']}\nAbstract: {row['Abstract']}\nKeywords: {row['Keywords']}\n"
        meta = {k: row[k] for k in ['EMDB_ID', 'Resolution', 'Method', 'Organisms', 'PDB_ID', 'Sample_Names', 'Publication_Title'] if k in row}
        documents.append(Document(content=content, meta=meta))

    doc_store.write_documents(documents)

    if output_file:
        # Optional: Save the document contents to a JSON for debugging or inspection
        with open(output_file, "w") as f:
            json.dump([{"content": d.content, "meta": d.meta} for d in documents], f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Haystack Documents from CSV input.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", required=False, help="Optional path to save documents as JSON.")

    args = parser.parse_args()
    main(args.input, args.output)
