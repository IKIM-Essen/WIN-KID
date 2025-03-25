import os
import sys
import argparse
import pandas as pd
from constants import NAMES_PATH
from constants import TRANSLATIONS_PATH
from fuzzywuzzy import process


def process_bvbcr(vitek_df, names_df, translations_df):
    # Group by genome ID
    for unique_genome_id in vitek_df["Genome ID"].unique():
        unique_genome_id_df = vitek_df[vitek_df["Genome ID"] == unique_genome_id]
        genome_transformed_df = pd.DataFrame(columns=["Sample_ID_IfH", "Organism_Code"])
        genome_transformed_df.at[0, "Sample_ID_IfH"] = unique_genome_id
        # TODO: What happens if multiple entries of the same GenomeID have different Names? Kommt das überhaupt vor?
        # Set name code
        name = unique_genome_id_df.iloc[0, 2]
        best_match_tuple = process.extractOne(
            name, names_df["Vitek_Name"], score_cutoff=80
        )
        if best_match_tuple:
            best_match = best_match_tuple[0]
            best_match_index = names_df[names_df["Vitek_Name"] == best_match].index[0]
        else:
            print(f"No names.csv match found for {name} and skipped instead")
            continue
        genome_transformed_df.at[0, "Organism_Code"] = names_df.iloc[
            best_match_index, 1
        ]
        print(genome_transformed_df)

    # Bring in right shape

    df = []
    return df


def load(input_path_load, output_path_load):

    # Ensure required files exist
    if not os.path.exists(NAMES_PATH):
        print(f"Error: You need to add a 'names.csv' file at {NAMES_PATH} to continue.")
        exit(1)

    if not os.path.exists(TRANSLATIONS_PATH):
        print(
            f"Error: You need to add a 'translations.csv' file at {TRANSLATIONS_PATH} to continue."
        )
        exit(1)

    # Ensure output directory exists
    os.makedirs(output_path_load, exist_ok=True)

    try:
        return (
            pd.read_csv(input_path_load, sep=";"),
            pd.read_csv(NAMES_PATH, sep=","),
            pd.read_csv(TRANSLATIONS_PATH, sep=","),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading CSV files: {e}")
        sys.exit(0)


if __name__ == "__main__":

    # LOAD
    parser = argparse.ArgumentParser(
        description="Parse BVBCR Vitek data and categorize bacteria."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output file")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    vitek_input, names_input, translations_input = load(input_path, output_path)

    # PROCESS
    processed_df = process_bvbcr(vitek_input, names_input, translations_input)

    # SAVE
