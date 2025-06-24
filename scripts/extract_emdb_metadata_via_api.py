import requests
import csv
import argparse
import os

def get_emdb_metadata(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        entry = response.json()

        if entry:
            emdb_id = entry.get("emdb_id", "")
            em_method = entry.get("structure_determination_list", {}).get("structure_determination", [{}])[0].get(
                "method", "")
            resolution = entry.get("structure_determination_list", {}).get("structure_determination", [{}])[0] \
                .get("image_processing", [{}])[0].get("final_reconstruction", {}).get("resolution", {}) \
                .get("valueOf_", "")
            pdb_ids = ",".join(
                ref.get("pdb_id", "")
                for ref in entry.get("crossreferences", {}).get("pdb_list", {}).get("pdb_reference", [])
            )
            organisms = ",".join(
                mm.get("natural_source", {}).get("organism", {}).get("valueOf_", "")
                for mm in entry.get("sample", {}).get("macromolecule_list", {}).get("macromolecule", [])
            )
            title = entry.get("admin", {}).get("title", "")
            keywords = ""  # not present in this structure
            publication = entry.get("crossreferences", {}).get("citation_list", {}) \
                .get("primary_citation", {}).get("citation_type", {}).get("title", "")
            sample_names = entry.get("sample", {}).get("name", {}).get("valueOf_", "")
            doi = ""
            pubmed = ""
            for ref in entry.get("crossreferences", {}).get("citation_list", {}) \
                    .get("primary_citation", {}).get("citation_type", {}).get("external_references", []):
                if ref.get("type_") == "DOI":
                    doi = ref.get("valueOf_", "").replace("doi:", "")
                if ref.get("type_") == "PUBMED":
                    pubmed = ref.get("valueOf_", "")
            abstract = entry.get("map", {}).get("annotation_details", "")

            return {
                "EMDB_ID": emdb_id,
                "EM_Method": em_method,
                "Resolution": resolution,
                "PDB_ID": pdb_ids,
                "Organisms": organisms,
                "Title": title,
                "Keywords": keywords,
                "Publication_Title": publication,
                "Sample_Names": sample_names,
                "DOI": doi,
                "Pubmed": pubmed,
                "Abstract": abstract
            }

    except Exception as e:
        print(f"Error parsing entry {entry.get('emdb_id', 'unknown')}: {e}")
        return None


def write_metadata_to_tsv(metadata_list, output_file):
    headers = [
        "EMDB_ID", "EM_Method", "Resolution", "PDB_ID", "Organisms",
        "Title", "Keywords", "Publication_Title", "Sample_Names",
        "DOI", "Pubmed", "Abstract"
    ]

    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        for row in metadata_list:
            writer.writerow(row)

def read_ids_from_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Fetch EMDB metadata and write to a TSV file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--id", type=str, help="Single EMDB ID (e.g. EMD-1010)")
    group.add_argument("-f", "--file", type=str, help="Path to a file containing EMDB IDs (one per line)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output TSV file name")

    args = parser.parse_args()

    if args.id:
        emdb_ids = [args.id]
    elif args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Input file {args.file} does not exist.")
            return
        emdb_ids = read_ids_from_file(args.file)

    metadata_list = []
    for emdb_id in emdb_ids:
        print(f"[INFO] Fetching metadata for {emdb_id}...")
        metadata = get_emdb_metadata(emdb_id)
        if metadata:
            metadata_list.append(metadata)

    if metadata_list:
        write_metadata_to_tsv(metadata_list, args.output)
        print(f"[INFO] Metadata written to {args.output}")
    else:
        print("[WARNING] No valid metadata retrieved.")

if __name__ == "__main__":
    main()
