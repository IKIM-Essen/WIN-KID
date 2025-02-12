# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.


import pandas as pd
import argparse

def txt_to_gff3(input_file, output_file):
    # Load the tab-separated TXT file
    df = pd.read_csv(input_file, sep="\t", dtype=str)
    print(df.iloc[:, :8])
    # Select and rename the required columns for GFF3 format
    df_gff = pd.DataFrame()
    df_gff["SeqID"] = df["ORF_ID"]  # Use ORF_ID as SeqID; sollte nach der ID abgeschnitten werden?
    df_gff["Source"] = "CARD" #erfüllt das unsere ansprüche?
    df_gff["Type"] = "gene"
    df_gff["Start"] = df["Start"].fillna(".")
    df_gff["End"] = df["Stop"].fillna(".")
    df_gff["Score"] = df["Pass_Bitscore"].fillna(".") #oder Best_Hit_Bitscore?
    df_gff["Strand"] = df["Orientation"].fillna(".")
    df_gff["Phase"] = "."

    # Construct the attributes column
    df_gff["Attributes"] = "ID=" + df["ORF_ID"] + ";"
    df_gff["Attributes"] += "Name=" + df["Best_Hit_ARO"] + ";"
    df_gff["Attributes"] += "DrugClass=" + df["Drug Class"].fillna("").str.replace(";", ",") + ";"
    df_gff["Attributes"] += "ResistanceMechanism=" + df["Resistance Mechanism"].fillna("") + ";"

    # Replace NaNs and ensure proper formatting
    df_gff.fillna(".", inplace=True)
    # Add GFF3 header
    with open(output_file, "w") as f:
        f.write("##gff-version 3\n")
        df_gff.to_csv(f, sep="\t", header=False, index=False)
        # Print completion message
        print("Conversion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a TXT file to GFF3 format.")
    parser.add_argument("input_file", help="Path to the input TXT file")
    parser.add_argument("output_file", help="Path to the output GFF3 file")
    args = parser.parse_args()

    txt_to_gff3(args.input_file, args.output_file)
