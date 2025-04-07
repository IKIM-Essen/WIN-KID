# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.


import argparse
import pandas as pd
import os
from constants import GFF_COLUMNS


def txt_to_gff3(df):
    # Split ORF_ID into ID and description
    df[["ORF_ID", "ORF_Desc"]] = df["ORF_ID"].str.extract(
        r"^(\S+)\s*(.*)$", expand=True
    )

    # Select and rename the required columns for GFF3 format
    df_gff = pd.DataFrame()
    df_gff[GFF_COLUMNS[0]] = df["ORF_ID"]  # Extracted ID only
    df_gff[GFF_COLUMNS[1]] = "CARD"
    df_gff[GFF_COLUMNS[2]] = "gene"
    df_gff[GFF_COLUMNS[3]] = df["Start"].fillna(".")
    df_gff[GFF_COLUMNS[4]] = df["Stop"].fillna(".")
    df_gff[GFF_COLUMNS[5]] = df["Best_Hit_Bitscore"].fillna(".")
    df_gff[GFF_COLUMNS[6]] = df["Orientation"].fillna(".")
    df_gff[GFF_COLUMNS[7]] = "."

    # Construct the attributes column with all required fields
    df_gff[GFF_COLUMNS[8]] = "Name=" + df["Best_Hit_ARO"].fillna("") + ";"
    df_gff[GFF_COLUMNS[8]] += (
        "DrugClass=" + df["Drug Class"].fillna("").str.replace(";", ",") + ";"
    )
    df_gff[GFF_COLUMNS[8]] += (
        "ResistanceMechanism=" + df["Resistance Mechanism"].fillna("") + ";"
    )
    df_gff[GFF_COLUMNS[8]] += (
        "AMRGeneFamily=" + df["AMR Gene Family"].fillna("").str.replace(";", ",") + ";"
    )
    df_gff[GFF_COLUMNS[8]] += (
        "Antibiotic=" + df["Antibiotic"].fillna("").str.replace(";", ",") + ";"
    )
    df_gff[GFF_COLUMNS[8]] += "ORF=" + df["ORF_Desc"].fillna("") + ";"

    # Replace NaNs with "."
    df_gff.fillna(".", inplace=True)

    return df_gff


if __name__ == "__main__":
    # LOAD
    parser = argparse.ArgumentParser(
        description="Convert all TXT files in a folder to GFF3 format."
    )
    parser.add_argument("input_folder", help="Path to the folder containing TXT files")
    parser.add_argument(
        "output_folder", help="Path to the folder for GFF3 output files"
    )
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Process all TXT files in the input folder
    txt_files = [f for f in os.listdir(args.input_folder) if f.endswith(".txt")]

    if not txt_files:
        print("No TXT files found in the input folder.")
        exit(1)

    for txt_file in txt_files:
        input_path = os.path.join(args.input_folder, txt_file)
        output_file = os.path.splitext(txt_file)[0] + ".gff"
        output_path = os.path.join(args.output_folder, output_file)

        # Load the TXT file
        df = pd.read_csv(input_path, sep="\t", dtype=str)

        # Process
        df_converted = txt_to_gff3(df)

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("##gff-version 3\n")
            df_converted.to_csv(f, sep="\t", header=False, index=False)

        print(f"Converted: {txt_file} → {output_file}")

    print(
        f"All TXT files in '{args.input_folder}' have been processed and saved to '{args.output_folder}'."
    )
