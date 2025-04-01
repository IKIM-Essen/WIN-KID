# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
from constants import NAMES_PATH
from constants import TRANSLATIONS_PATH
from fuzzywuzzy import process


def process_bvbcr(vitek_df, names_df, translations_df):

    df = pd.DataFrame()
    for unique_genome_id in vitek_df["Genome ID"].unique():
        # Group by genome ID
        unique_genome_id_df = vitek_df[vitek_df["Genome ID"] == unique_genome_id]
        genome_transformed_df = pd.DataFrame(columns=["Sample_ID_IfH", "Organism_Code"])
        genome_transformed_df.at[0, "Sample_ID_IfH"] = unique_genome_id

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

        # Add antibiotics
        for _, row in unique_genome_id_df.iterrows():
            if row["Measurement Sign"] != "":
                measurement = (
                    row["Measurement"]
                    .replace("<", "")
                    .replace(">", "")
                    .replace("=", "")
                )
                measurement = (
                    str(row["Measurement Sign"]).replace("nan", "") + measurement
                )
            else:
                measurement = row["Measurement"]
            genome_transformed_df[row["Antibiotic"]] = measurement
        df = pd.concat([df, genome_transformed_df], ignore_index=True)
        df = translate(df, translations_df)
    return df


def translate(input_df, translations):
    rename_dict = dict(zip(translations["Old"], translations["New"]))
    for old, new in rename_dict.items():
        input_df.columns = input_df.columns.str.replace(old, new, regex=True)
    return input_df


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
    os.makedirs(Path(output_path_load).parent, exist_ok=True)

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
    processed_df.to_csv(output_path, index=False)
    print(f"Cleaned bvbcr file saved to: {output_path}")
