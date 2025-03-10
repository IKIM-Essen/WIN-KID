# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import re
import os
import sys
import argparse
from pprint import pprint
import pandas as pd
from fuzzywuzzy import fuzz


def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^a-zA-Z\s]", "", text).lower()


def clean_dataframe(input_df, bacteria, code):
    print(f"Keys used: {bacteria}")
    input_df = input_df[input_df["ERREGERLANG"].isin(bacteria)]
    df = pd.DataFrame(columns=["LABORNR"])

    for _, row in input_df.iterrows():
        if str(row["ANTIBIOTIKA"]) == "nan":
            continue
        if "MRGN" in row["ANTIBIOTIKA"]:
            continue

        if row["LABORNR"] in df["LABORNR"].tolist():
            index = df.loc[df["LABORNR"] == row["LABORNR"]].index[0]
        else:
            index = len(df)
            df.at[index, "LABORNR"] = row["LABORNR"]

        value = (str(row["MHK-VKZ"]) + str(row["MHK-Wert"])).replace("nan", "")

        if value != "":
            df.at[
                index,
                row["ANTIBIOTIKA"]
                .split("(", 1)[0]
                .replace("/", "-")
                .replace("+", "-")
                .replace("_", "-")
                .replace(" ", ""),
            ] = value

    df.insert(loc=1, column="Organism_Code", value=code)
    df = df.dropna(axis=0, how="all", subset=df.columns[2:])
    df = df.replace("NA", None)
    df = df.dropna(axis=1, how="all")
    df = df.fillna("NA")
    df = df.rename(columns={"LABORNR": "Sample_ID_IfH"})
    return df


def assign_unique(df, categories):
    unique_bacteria = df["ERREGERLANG"].dropna().unique()
    output = {c: [] for c in categories}

    for b in unique_bacteria:
        scores = []
        for c in categories:
            similarity = fuzz.token_set_ratio(clean_text(b), clean_text(c))
            scores.append(similarity)
        highest_index = scores.index(max(scores))
        output[categories[highest_index]].append(b)

    pprint(output)

    return output


def translate(input_df, translations):
    rename_dict = dict(zip(translations["Old"], translations["New"]))
    for old, new in rename_dict.items():
        input_df.columns = input_df.columns.str.replace(old, new, regex=True)
    return input_df


# execution in terminal
if __name__ == "__main__":

    # LOAD
    parser = argparse.ArgumentParser(
        description="Parse Vitek data and categorize bacteria."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    INPUT_PATH = args.input_file
    OUTPUT_FOLDER = args.output_dir

    # paths to required files (static)
    NAMES_PATH = "resources/settings/names.csv"
    TRANSLATIONS_PATH = "resources/settings/translations.csv"

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
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        vitek_df = pd.read_csv(INPUT_PATH, sep=",")
        names_df = pd.read_csv(NAMES_PATH, sep=",")
        translations_df = pd.read_csv(TRANSLATIONS_PATH, sep=",")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading CSV files: {e}")
        sys.exit(0)

    # PROCESS
    assignments = assign_unique(vitek_df, names_df["Vitek_Name"])

    cleaned_df_dic = {}
    for name in names_df["Vitek_Name"]:
        cleaned_df = clean_dataframe(
            vitek_df,
            assignments[name],
            names_df.loc[names_df["Vitek_Name"] == name, "Code"].values[0],
        )
        cleaned_df = translate(cleaned_df, translations_df)
        cleaned_df_dic[name] = cleaned_df

    # SAVE
    for name in names_df["Vitek_Name"]:
        cleaned_df_dic[name].to_csv(
            os.path.join(OUTPUT_FOLDER, f"{name.lower().replace(' ', '_')}.csv"),
            index=False,
        )
        print(
            f"All {name} saved to {
                os.path.join(OUTPUT_FOLDER, name.lower().replace(' ', '_') + '.csv')
                }"
        )
