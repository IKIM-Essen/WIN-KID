# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import argparse
import sys
import os
import pandas as pd
from constants import TRANSLATIONS_PATH


def process(df, translations_df):
    df = df.loc[:, ~df.columns.str.contains("Family|Andere-Instrument", regex=True)]
    df.columns = df.columns.str.strip()
    columns_to_drop = [
        "Card_Name",
        "Lab_ID",
        "Isolate_Number",
        "Patient_ID",
        "Specimen_Source",
        "Collection_Date.x",
        "Testing_Date",
        "Organism_Name",
        "Card_Type",
        "Lot_Number",
        "Expiration_Date",
        "Bar_Code",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df = df.rename(
        columns={
            col: col.split("-", 1)[1] if "-" in col else col for col in df.columns[2:]
        }
    )
    df = df.replace(",", ".", regex=True)
    df.iloc[:, 2:] = df.iloc[:, 2:].map(
        lambda x: (
            "NA" if isinstance(x, str) and not any(char.isdigit() for char in x) else x
        )
    )
    df.columns = [df.columns[0], df.columns[1]] + [
        col.split("(", 1)[0]
        .replace("/", "-")
        .replace("+", "-")
        .replace("_", "-")
        .replace(" ", "")
        for col in df.columns[2:]
    ]

    df = df.replace("NA", None)
    df = df.dropna(axis=1, how="all")

    # Replace missing values with 'NA'
    df = df.fillna("NA")

    translate(df, translations_df)

    return df


def translate(input_df, translations):
    rename_dict = dict(zip(translations["Old"], translations["New"]))
    for old, new in rename_dict.items():
        input_df.columns = input_df.columns.str.replace(old, new, regex=True)
    return input_df


# Execution in terminal
if __name__ == "__main__":
    # PROCESS
    parser = argparse.ArgumentParser(
        description="Parse Vitek data and categorize bacteria."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "output_file", help="Path to output CSV file"
    )  # Changed to "file"
    args = parser.parse_args()

    INPUT_PATH = args.input_file
    OUTPUT_PATH = args.output_file

    # Ensure OUTPUT_PATH is a valid file path, not a directory
    if os.path.isdir(OUTPUT_PATH):
        print(
            f"Error: '{OUTPUT_PATH}' is a directory. Please provide a valid file path."
        )
        sys.exit(0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Loading data from: {INPUT_PATH}")
    try:
        vitek_df = pd.read_csv(INPUT_PATH, sep="\t", quotechar='"')
        translations_input = pd.read_csv(TRANSLATIONS_PATH, sep=",")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading CSV files: {e}")
        sys.exit(0)

    # PROCESS
    cleaned_df = process(vitek_df, translations_input)
    print(cleaned_df.head())  # Show first few rows to check if processing worked

    # SAVE
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned file saved to: {OUTPUT_PATH}")
