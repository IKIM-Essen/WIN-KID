# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import pandas as pd
import argparse


def clean_dataframe(df):
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = [df.columns[0], df.columns[1]] + [
        col.split("(", 1)[0]
        .replace("/", "-")
        .replace("+", "-")
        .replace("_", "-")
        .replace(" ", "")
        for col in df.columns[2:]
    ]
    return df

# Execution in terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Vitek data and categorize bacteria.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")  # Changed to "file"
    args = parser.parse_args()

    INPUT_PATH = args.input_file
    OUTPUT_PATH = args.output_file

    # Paths to required files (static)
    TRANSLATIONS_PATH = "resources/translations.csv"

    # Ensure required files exist
    if not os.path.exists(TRANSLATIONS_PATH):
        print(f"Error: 'translations.csv' not found at {TRANSLATIONS_PATH}.")
        exit(1)
    
    # Ensure OUTPUT_PATH is a valid file path, not a directory
    if os.path.isdir(OUTPUT_PATH):
        print(f"Error: '{OUTPUT_PATH}' is a directory. Please provide a valid file path.")
        exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Loading data from: {INPUT_PATH}")
    try:
        vitek_df = pd.read_csv(INPUT_PATH, sep="\t", quotechar='"')
        translations_df = pd.read_csv(TRANSLATIONS_PATH, sep=",")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        exit(1)

    # Clean
    cleaned_df = clean_dataframe(vitek_df)
    print(cleaned_df.head())  # Show first few rows to check if processing worked

    # Save
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned file saved to: {OUTPUT_PATH}")
