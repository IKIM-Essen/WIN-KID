# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from enum import Enum
import pandas as pd
from fuzzywuzzy import fuzz
import os


class EucastInterpretation(Enum):
    S = 1
    I = 2
    R = 3


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def confirm(question):
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer == "y":
            return True
        elif answer == "n":
            return False
        else:
            print("Invalid Input")


# get best option from eucast matches
def get_most_similar_name(input_df, target_name, cut_off):
    input_df = input_df.copy()
    input_df["Name"] = input_df["Name"].str.split("(", n=1).str[0].str.strip()
    input_df["similarity_score"] = input_df["Name"].apply(
        lambda name: fuzz.ratio(target_name, name)
    )
    best_match_index = input_df["similarity_score"].idxmax()
    best_match = input_df.loc[[best_match_index]].copy()
    if best_match["similarity_score"].iloc[0] < cut_off:
        # only give console output if target_name isn't ignored
        if not target_name in ignore_df["Ignorelist"].tolist():
            print(f"No EUCAST Match with Score >= {cut_off}: {target_name}")
        return pd.DataFrame()

    return best_match


# handle no eucast match (translate or ignore)
def handle_no_match(input, eucast, ignore_df, translations_df):
    print(f"Handling no match for {input}...")
    best_match = get_most_similar_name(eucast, input, 0)
    print(f"Best match: {best_match['Name'].iloc[0]}")
    use_match = confirm("Do you want to use it in the future?")
    if use_match:
        if not input in translations_df["Old"].tolist():
            translations_df.loc[len(translations_df)] = [
                input,
                best_match["Name"].iloc[0],
            ]
        else:
            print(f"{input} already translated, run the parser again.")
    else:
        ignore = confirm(f"Do you want to ignore {input}?")
        if ignore:
            ignore_df.loc[len(ignore_df), "Ignorelist"] = input


# interpret vitek value with eucast
def get_mic_interpretation(columns_vitek, rows_eucast, antibiotic_name_vitek, df):
    index = 0
    for data in columns_vitek:
        interpretation = ""
        # handle 'NA's
        if isinstance(data, float):
            interpretation = "NA"
        # vitek interpretation
        else:
            data = data.replace(">", "").replace("=", "").replace(",", ".")
            if "<" in data:
                interpretation = EucastInterpretation(1).name
            elif float(data) <= float(rows_eucast["S <="].iloc[0]):
                interpretation = EucastInterpretation(1).name
            elif float(data) <= float(rows_eucast["R >"].iloc[0]):
                interpretation = EucastInterpretation(2).name
            else:
                interpretation = EucastInterpretation(3).name

        df.at[index, antibiotic_name_vitek] = interpretation
        index += 1
    return df


# interpret whole vitek table
def interpret_vitek(
    input_vitek, input_eucast, handle_no_matches, ignore_df, translations_df
):
    output_df = input_vitek[["LABORNR", "Organism_Code"]].copy()
    # Clean VITEK data
    input_vitek.columns = input_vitek.columns.str.strip()
    input_vitek = input_vitek.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Process each antibiotic column
    for column_vitek in input_vitek.columns[2:]:
        # Adapt VITEK name to EUCAST
        column_name_vitek = (
            column_vitek.split("(", 1)[0]
            .replace("/", "-")
            .replace("+", "-")
            .replace(" ", "")
        )

        matching_rows_eucast = input_eucast.loc[
            input_eucast["Name"].str.contains(column_name_vitek, case=False, na=False)
        ]

        # Remove faulty matches
        removed_match = False
        for index, row in matching_rows_eucast.iterrows():
            if (not is_float(row["S <="])) or (not is_float(row["R >"])):
                matching_rows_eucast = matching_rows_eucast.drop(index)
                removed_match = True

        # Get best match from found matches
        if not matching_rows_eucast.empty:
            matching_rows_eucast = get_most_similar_name(
                matching_rows_eucast, column_name_vitek, 70
            )
            # if no match is above 70% cutoff
            if matching_rows_eucast.empty and handle_no_matches:
                handle_no_match(
                    column_name_vitek, input_eucast, ignore_df, translations_df
                )
        # if no match was found and vitek name isnt in ignorelist
        elif (
            not removed_match
            and not column_name_vitek in ignore_df["Ignorelist"].tolist()
        ):
            print(f"No EUCAST Match: {column_name_vitek}")
            if handle_no_matches:
                handle_no_match(
                    column_name_vitek, input_eucast, ignore_df, translations_df
                )
        if matching_rows_eucast.empty:
            continue

        # Interpret Match
        column_data_vitek = input_vitek[column_vitek]
        output_df = get_mic_interpretation(
            column_data_vitek, matching_rows_eucast, column_name_vitek, output_df
        )

    return output_df


# Paths
INPUT_VITEK_PATH = "output/vitek_parsed/"
NAMES_PATH = "resources/names.csv"
IGNORE_PATH = "resources/ignore.csv"
TRANSLATIONS_PATH = "resources/translations.csv"

# Load
names_df = pd.read_csv(NAMES_PATH)
ignore_df = pd.read_csv(IGNORE_PATH)
translations_df = pd.read_csv(TRANSLATIONS_PATH)

# Toggle if no eucast matches should be handled
HANDLE_NO_MATCHES = False

# Interpret & Save
for vitek_file_name in os.listdir(INPUT_VITEK_PATH):
    if vitek_file_name.endswith(".csv"):
        input_vitek = pd.read_csv(INPUT_VITEK_PATH + vitek_file_name)
        eucast_file_name = names_df.loc[
            names_df["Code"] == input_vitek["Organism_Code"].iloc[0], "Eucast_File_Name"
        ].values[0]
        input_eucast = pd.read_json(f"resources/{eucast_file_name}")
        output_df = interpret_vitek(
            input_vitek, input_eucast, HANDLE_NO_MATCHES, ignore_df, translations_df
        )
        output_df.to_csv(f"output/interpreted/{vitek_file_name}", index=False)
        print(f"Cleaned file saved to: output/interpreted/{vitek_file_name}")

ignore_df.to_csv(IGNORE_PATH, index=False)
translations_df.to_csv(TRANSLATIONS_PATH, index=False)
