# EMSummary

**EMSummary** is a Python-based pipeline for generating concise, human-readable summaries of cryo-EM structures archived in the [Electron Microscopy Data Bank (EMDB)](https://www.ebi.ac.uk/emdb/). It leverages natural language processing (NLP) techniques to assist researchers in quickly grasping the essential metadata associated with EMDB entries.

## 🚀 Features

- Extracts and processes key metadata fields from EMDB XML files (e.g., method, resolution, sample, title, abstract, keywords).
- Generates summaries using:
  - Extractive summarization with `facebook/bart-large-cnn`
- Evaluates generated summaries using:
  - ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
  - BERTScore (precision, recall, F1)
- Outputs results as structured `.tsv` files.


## 🛠️ Installation

We recommend using a Python virtual environment.

```bash
git clone https://github.com/amudha-kumari/EMSummary.git
cd EMSummary
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


📥 Input Data

    You can download EMDB XML entries from the EMDB FTP archive.

    Save the XML files in the data/emdb_entries/ folder.

🔄 Usage
1. Extract Metadata

python scripts/extract_emdb_metadata.py --xml_dir /path/to/xml_files --out_file out_file_name.tsv

2. Generate Summaries using Huggigface BART

python scripts/generate_summaries_with_BART.py --in_file /path/to/metadata_file --out_file out_file.tsv

3.Generate fine-tuned models for EMDB-specific metadata

python scripts/create_finetuned_models.py --metadata path/to/metadata.tsv --summaries path/to/summaries.tsv --out_dir path/to/save_model


4. Generate Summaries with fine-tuned models

python scripts/generate_finetuned_summaries.py --model_dir /path/to/model_dir/ --input_tsv input_metadata_file --output_tsv out_file.tsv --num_entries number_of_emdb_entries

5. Evaluate Summaries

python scripts/evaluate_summary_quality.py

    ⚠️ The evaluate_summaries.py script requires both EM_input.tsv (with reference text) and EM_summaries.tsv (with generated summaries) to be present in the data/ folder.

📊 Example Output

The final evaluation output (evaluated_summaries.tsv) includes:
Entry_No	EMDB_ID	ROUGE-1	ROUGE-2	ROUGE-L	BERTScore-F1
1	EMD-38755	0.716	0.709	0.716	0.898
2	EMD-38745	0.397	0.367	0.397	0.893

📬 Contact

For questions or contributions, please contact Amudha Kumari Duraisamy or open an issue on this repository.

