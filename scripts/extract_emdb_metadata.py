"""
extract_emdb_metadata.py

Extracts structured metadata from EMDB XML files into a tabular format (TSV or CSV)
for downstream processing and analysis.

The script parses EMDB XML entries to retrieve relevant metadata fields such as:
- EMDB ID, Electron microscopy method, Resolution, Title, Keywords, Publication title, Sample description, DOI, PubMed ID, Abstract

These fields are used to support natural language processing tasks such as
automatic summary generation and metadata enrichment for EMDB entries.

Usage:
    python extract_emdb_metadata.py --xml_dir /path/to/xml_files --out_file out_file_name.tsv

Arguments:
    --xml_dir      Path to the directory containing EMDB XML files.
    --out_file       Output file path for the extracted metadata (TSV or CSV).
    --format       (Optional) Output format: 'tsv' (default) or 'csv'.

Dependencies:
    - Python 3.x
    - xml.etree.ElementTree (standard library)
    - pandas

Author:
    Dr. rer. nat. Amudha Kumari Duraisamy
email:
    amudhakumari@gmail.com
"""

import os
import argparse
import xml.etree.ElementTree as ET
import csv
import requests
import re


# Function to clean and process text
def clean_text(text):
    if not text:
        return ""

    # Remove XML/HTML tags (if any slipped in)
    text = re.sub(r'<[^>]+>', '', text)

    # Replace multiple whitespace/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-breaking spaces and other non-ASCII chars if needed
    text = text.replace('\xa0', ' ').strip()

    return text


# Function to fetch abstract from Europe PMC using DOI or PubMed ID
def fetch_abstract(doi=None, pmid=None):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    if not (doi or pmid):
        return None  # If neither DOI nor PubMed ID is provided, return None

    query = f'DOI:"{doi}"' if doi else f'EXT_ID:{pmid} AND SRC:MED'

    params = {
        'query': query,
        'format': 'json',
        'pageSize': 1,
        'resultType': 'core'  # Request full metadata including the abstract
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        result = response.json()

        if 'resultList' in result and result['resultList']['result']:
            abstract = result['resultList']['result'][0].get('abstractText', None)
            if abstract:
                # Clean the abstract
                return clean_text(abstract)

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch record data: {e}")

    return None


# Function to extract metadata from an XML file
def extract_metadata(xml_file):
    """Extracts required metadata from an XML file."""
    metadata = {
        "EMDB ID": os.path.basename(xml_file).split('.')[0].replace("emd-", "EMD-").replace("-v30", ""),
        "Method": "Unknown",
        "Resolution": "Unknown",
        "PDB ID": "Unknown",
        "Organisms": "Unknown",
        "Title": "Unknown",
        "Keywords": [],
        "Publication Title": "Unknown",
        "Sample Names": "Unknown",
        "DOI": "Unknown",
        "Pubmed": "Unknown",
        "Abstract": "Unknown"
    }

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Method and Resolution extraction
        method = root.findtext(".//structure_determination/method")
        resolution = None

        method_mapping = {
            "singleParticle": "Single Particle Analysis",
            "tomography": "Tomography",
            "subtomographyAveraging": "SubTomography Averaging",
            "Helical": "Helical",
            "electronCrystallography": "Electron Crystallography"
        }

        if method in method_mapping:
            metadata["Method"] = method_mapping[method]
            if method == "subtomographyAveraging":
                method_path = "subtomogram_averaging_preparation"
            elif method == "electronCrystallography":
                method_path = "crystallography_preparation"
            else:
                method_path = method

            resolution_path = f".//structure_determination_list//structure_determination/{method_path.lower()}_processing/final_reconstruction/resolution"
            resolution = root.findtext(resolution_path)


        if resolution:
            metadata["Resolution"] = resolution.strip()

        # Title
        title = root.findtext(".//admin/title")
        if title:
            metadata["Title"] = clean_text(title)

        # Keywords
        keywords_text = root.findtext(".//admin/keywords")
        if keywords_text:
            metadata["Keywords"] = [clean_text(kw.strip()) for kw in keywords_text.split(',')] if keywords_text else []

        # Publication Title
        publication_title = root.findtext(".//crossreferences/citation_list/primary_citation/journal_citation/title")
        if publication_title:
            metadata["Publication Title"] = clean_text(publication_title)

        # DOI and PubMed
        external_refs = root.findall(
            ".//crossreferences/citation_list/primary_citation/journal_citation/external_references")
        for ref in external_refs:
            ref_type = ref.attrib.get("type", "").upper()
            ref_value = ref.text.strip() if ref.text else ""

            if ref_type == "DOI":
                metadata["DOI"] = ref_value.replace("doi:", "").strip()
                metadata["Abstract"] = fetch_abstract(doi=metadata["DOI"])
            elif ref_type == "PUBMED":
                metadata["Pubmed"] = ref_value
                metadata["Abstract"] = fetch_abstract(pmid=metadata["Pubmed"])

        # # Sample (quoted)
        # sample = root.findtext(".//sample/name")
        # if sample:
        #     metadata["Sample"] = clean_text(sample)

        nat_source = root.findall(".//natural_source/organism")
        unique_organisms = sorted({org.text for org in nat_source if org.text})
        metadata["Organisms"] = [", ".join(unique_organisms)]

        pdb_id = root.findtext(".//crossreferences/pdb_list/pdb_reference/pdb_id")
        metadata["PDB ID"] = pdb_id if pdb_id else "Unknown"

        sample_name = root.findall(".//name")
        names = sorted({elem.text for elem in sample_name if elem.text})
        metadata["Sample Names"] =[", ".join(names)]


    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
        return None

    return metadata


# Function to extract metadata from all XML files in a folder
def extract_from_xml_files(folder_path, out_file=None):
    meta_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".xml"):
            file_path = os.path.join(folder_path, file)
            metadata = extract_metadata(file_path)
            if metadata:
                metadata["Keywords"] = str(metadata["Keywords"])  # Ensure Keywords is serialized as a list string
                meta_data.append([metadata[h] for h in [
                    "EMDB ID", "Method", "Resolution", "PDB ID", "Organisms", "Title", "Keywords",
                    "Publication Title", "Sample Names", "DOI", "Pubmed", "Abstract"
                ]])

    # Define headers
    headers = ["EMDB_ID", "EM_Method", "Resolution", "PDB_ID", "Organisms", "Title", "Keywords",
               "Publication_Title", "Sample_Names", "DOI", "Pubmed", "Abstract"]

    # Write to TSV file (overwrite if it exists)
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(output_folder, exist_ok=True)

    out_file = os.path.join(output_folder, out_file)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(meta_data)

    print(f"Metadata saved to: {out_file}")


# Main function to run the extraction process
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract metadata from EMDB XML files.")
    parser.add_argument('--xml_dir', type=str, required=True, help="Path to the directory containing EMDB XML files.")
    parser.add_argument('--out_file', type=str, required=True,
                        help="Output file name for the extracted metadata (TSV or CSV).")

    args = parser.parse_args()

    # Validate the input directory
    if not os.path.exists(args.xml_dir):
        print(f"Directory '{args.xml_dir}' not found.")
        return

    # Call the extraction function with the provided arguments
    extract_from_xml_files(args.xml_dir, args.out_file)


if __name__ == "__main__":
    main()
