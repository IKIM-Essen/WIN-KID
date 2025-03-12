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


def process(vitek_df, names_df, translations_df):
    assignments = assign_unique(vitek_df, names_df["Vitek_Name"])

    cleaned_df_dic = {}
    for process_name in names_df["Vitek_Name"]:
        cleaned_df = clean_dataframe(
            vitek_df,
            assignments[process_name],
            names_df.loc[names_df["Vitek_Name"] == process_name, "Code"].values[0],
        )
        cleaned_df = translate(cleaned_df, translations_df)
        cleaned_df_dic[process_name] = cleaned_df

    return cleaned_df_dic


def load(input_path_load, output_path_load):

    # paths to required files (static)
    names_path = "resources/settings/names.csv"
    translations_path = "resources/settings/translations.csv"

    # Ensure required files exist
    if not os.path.exists(names_path):
        print(f"Error: You need to add a 'names.csv' file at {names_path} to continue.")
        exit(1)

    if not os.path.exists(translations_path):
        print(
            f"Error: You need to add a 'translations.csv' file at {translations_path} to continue."
        )
        exit(1)

    # Ensure output directory exists
    os.makedirs(output_path_load, exist_ok=True)

    try:
        return (
            pd.read_csv(input_path_load, sep=","),
            pd.read_csv(names_path, sep=","),
            pd.read_csv(translations_path, sep=","),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading CSV files: {e}")
        sys.exit(0)


# execution in terminal
if __name__ == "__main__":

    # LOAD
    parser = argparse.ArgumentParser(
        description="Parse Vitek data and categorize bacteria."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_dir
    vitek_input, names_input, translations_input = load(input_path, output_path)

    # PROCESS
    processed_df = process(vitek_input, names_input, translations_input)

    # SAVE
    for name in names_input["Vitek_Name"]:
        processed_df[name].to_csv(
            os.path.join(output_path, f"{name.lower().replace(' ', '_')}.csv"),
            index=False,
        )
        print(
            f"All {name} saved to {os.path.join(output_path, name.lower().replace(' ', '_') + '.csv')}"
        )
