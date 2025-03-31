# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.


import argparse
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Convert a TXT file to GFF3 format.")
    parser.add_argument("input_file", help="Path to the input TXT file")
    parser.add_argument("output_file", help="Path to the output GFF3 file")
    args = parser.parse_args()
    # Load the tab-separated TXT file
    df = pd.read_csv(args.input_file, sep="\t", dtype=str)

    # PROCESS
    df_converted = txt_to_gff3(df)

    # SAVE
    with open(args.output_file, "w") as f:
        f.write("##gff-version 3\n")
        df_converted.to_csv(f, sep="\t", header=False, index=False)

    # Print completion message
    print(
        f"CARD txt to GFF3 conversion completed successfully! Output saved to: {args.output_file}"
    )
