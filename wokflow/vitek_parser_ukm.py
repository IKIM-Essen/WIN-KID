# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import pandas as pd


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


# Paths
INPUT_PATH = "resources/MHK_UKM_Subset.csv"
OUTPUT_PATH = "output/vitek_parsed/ukm_parsed.csv"
TRANSLATIONS_PATH = "resources/translations.csv"

# Load
vitek_df = pd.read_csv(INPUT_PATH, sep="\t", quotechar='"')
translations_df = pd.read_csv(TRANSLATIONS_PATH, sep=",")

# Clean
cleaned_df = clean_dataframe(vitek_df)

# create missing output directory
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Save
cleaned_df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned file saved to: {OUTPUT_PATH}")
