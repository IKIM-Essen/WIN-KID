# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
from enum import Enum
import pandas as pd
from fuzzywuzzy import fuzz
import argparse


class EucastInterpretation(Enum):
    S = 1
    I = 2
    R = 3
    NA = 4


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def confirm(question):
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ["y", "n"]:
            return answer == "y"
        print("Invalid Input")


def get_most_similar_name(input_df, target_name, cut_off):
    input_df = input_df.copy()
    input_df["Name"] = input_df["Name"].str.split("(", n=1).str[0].str.strip()
    input_df["similarity_score"] = input_df["Name"].apply(
        lambda name: fuzz.ratio(target_name, name)
    )
    best_match_index = input_df["similarity_score"].idxmax()
    best_match = input_df.loc[[best_match_index]].copy()
    if best_match["similarity_score"].iloc[0] < cut_off:
        if not target_name in ignore_df["Ignorelist"].tolist():
            print(f"No EUCAST Match with Score >= {cut_off}: {target_name}")
        return pd.DataFrame()

    return best_match


def handle_no_match(name, eucast):
    if name in translations_df["Old"].tolist():
        print(f"{name} already translated, rerun parser to load")
        return
    print(f"Handling no match for {name}...")
    best_match = get_most_similar_name(eucast, name, 0)
    print(f"Best match: {best_match['Name'].iloc[0]}")
    use_match = confirm("Do you want to use it in the future?")
    if use_match:
        if not name in translations_df["Old"].tolist():
            translations_df.loc[len(translations_df)] = [
                name,
                best_match["Name"].iloc[0],
            ]
        else:
            print(f"{name} already translated, run the parser again.")
    else:
        ignore_check = confirm(f"Do you want to ignore {name}?")
        if ignore_check:
            ignore_df.loc[len(ignore_df), "Ignorelist"] = name


def get_mic_interpretation(columns_vitek, rows_eucast, antibiotic_name_vitek, df):
    index = 0
    for data in columns_vitek:
        interpretation = ""
        if isinstance(data, float):
            interpretation = EucastInterpretation(4).name
        else:
            raw_data = float(
                data.replace(">", "")
                .replace("=", "")
                .replace("<", "")
                .replace(",", ".")
            )
            s_eucast = float(rows_eucast["S <="].iloc[0])
            r_eucast = float(rows_eucast["R >"].iloc[0])
            if raw_data == s_eucast and raw_data == r_eucast:
                if ">" in data:
                    interpretation = EucastInterpretation(3).name
                else:
                    interpretation = EucastInterpretation(1).name
            elif raw_data <= s_eucast:
                interpretation = EucastInterpretation(1).name
            elif raw_data > r_eucast:
                interpretation = EucastInterpretation(3).name
            else:
                interpretation = EucastInterpretation(2).name

        df.at[index, antibiotic_name_vitek] = interpretation
        index += 1
    return df


def interpret_vitek(input_vitek, input_eucast):
    df = input_vitek[["Sample_ID_IfH", "Organism_Code"]].copy()
    input_vitek.columns = input_vitek.columns.str.strip()
    input_vitek = input_vitek.map(lambda x: x.strip() if isinstance(x, str) else x)

    for column_vitek in input_vitek.columns[2:]:

        matching_rows_eucast = input_eucast.loc[
            input_eucast["Name"].str.contains(column_vitek, case=False, na=False)
        ]

        removed_match = False
        for index, row in matching_rows_eucast.iterrows():
            if (not is_float(row["S <="])) or (not is_float(row["R >"])):
                matching_rows_eucast = matching_rows_eucast.drop(index)
                removed_match = True

        if not matching_rows_eucast.empty:
            matching_rows_eucast = get_most_similar_name(
                matching_rows_eucast, column_vitek, 70
            )
            if (
                matching_rows_eucast.empty
                and HANDLE_NO_MATCHES
                and not column_vitek in ignore_df["Ignorelist"].tolist()
            ):
                handle_no_match(column_vitek, input_eucast)
        elif not removed_match and not column_vitek in ignore_df["Ignorelist"].tolist():
            print(f"No EUCAST Match: {column_vitek}")
            if HANDLE_NO_MATCHES:
                handle_no_match(column_vitek, input_eucast)
        if matching_rows_eucast.empty:
            continue

        column_data_vitek = input_vitek[column_vitek]
        df = get_mic_interpretation(
            column_data_vitek, matching_rows_eucast, column_vitek, df
        )

    return df


def interpret_folder(vitek_folder, output_folder):
    for vitek_file_name in os.listdir(vitek_folder):
        if vitek_file_name.endswith(".csv"):
            vitek_path = os.path.join(vitek_folder, vitek_file_name)
            output_path = os.path.join(
                output_folder, vitek_file_name.replace("parsed", "interpreted")
            )

            vitek_df = pd.read_csv(vitek_path)
            split_df = {key: group for key, group in vitek_df.groupby("Organism_Code")}
            output_df = pd.DataFrame()

            for key, df in split_df.items():
                matching_json_name = names_df.loc[
                    names_df["Code"] == key,
                    "Eucast_File_Name",
                ].values[0]
                print(f"Used {matching_json_name} for {os.path.basename(vitek_path)}")
                matching_json = pd.read_json(INPUT_EUCAST_FOLDER + matching_json_name)
                interpreted_df = interpret_vitek(df, matching_json)
                output_df = pd.concat(
                    [output_df, interpreted_df], ignore_index=True
                ).fillna("NA")

            output_df.to_csv(output_path, index=False)
            print(f"Interpreted file saved to: {output_path}")


# terminal input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Vitek data and categorize bacteria."
    )
    parser.add_argument(
        "input_folder", help="Path to input folder containing CSV files"
    )
    parser.add_argument(
        "output_folder", help="Path to output directory for interpreted files"
    )
    args = parser.parse_args()

    INPUT_VITEK_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder


# Paths (Misc.)
NAMES_PATH = "resources/names.csv"
IGNORE_PATH = "resources/ignore.csv"
TRANSLATIONS_PATH = "resources/translations.csv"
INPUT_EUCAST_FOLDER = "resources/eucast_files/"

# create missing output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load
names_df = pd.read_csv(NAMES_PATH)
ignore_df = pd.read_csv(IGNORE_PATH)
translations_df = pd.read_csv(TRANSLATIONS_PATH)

# Toggle if no eucast matches should be handled
HANDLE_NO_MATCHES = False

# Interpret & Save
interpret_folder(INPUT_VITEK_FOLDER, OUTPUT_FOLDER)

ignore_df.to_csv(IGNORE_PATH, index=False)
translations_df.to_csv(TRANSLATIONS_PATH, index=False)
