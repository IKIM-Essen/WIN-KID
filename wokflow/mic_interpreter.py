"""
Test
"""

# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
from enum import Enum
import pandas as pd
from fuzzywuzzy import fuzz


class EucastInterpretation(Enum):
    """Match Number with R/I/S"""

    S = 1
    I = 2
    R = 3


def is_float(value):
    """Check if something can be converted to a float"""
    try:
        float(value)
        return True
    except ValueError:
        return False


def confirm(question):
    """Ask user to confirm something"""
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ["y", "n"]:
            return answer == "y"
        print("Invalid Input")


def get_most_similar_name(input_df, target_name, cut_off):
    """Get best option from Eucast matches"""
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


def handle_no_match(name, eucast):
    """Handle no eucast match (translate or ignore)"""
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
    """Interpret Mic"""
    index = 0
    for data in columns_vitek:
        interpretation = ""
        # handle 'NA's
        if isinstance(data, float):
            interpretation = "NA"
        # MIC interpretation
        else:
            raw_data = float(
                data.replace(">", "")
                .replace("=", "")
                .replace("<", "")
                .replace(",", ".")
            )
            s_eucast = float(rows_eucast["S <="].iloc[0])
            r_eucast = float(rows_eucast["R >"].iloc[0])
            interpretations = []
            if raw_data <= s_eucast:
                interpretations.append("S")
            if raw_data >= r_eucast:
                interpretations.append("R")
            if len(interpretations) == 0:
                interpretation = "I"
            elif len(interpretations) == 1:
                interpretation = interpretations[0]
            else:
                if "<" in data:
                    interpretation = "S"
                elif ">" in data:
                    interpretation = "R"
                else:
                    interpretation = "I"
                print(interpretations, data)

            # if "<" in data:
            #     interpretation = EucastInterpretation(1).name
            # elif float(data) <= float(rows_eucast["S <="].iloc[0]):
            #     interpretation = EucastInterpretation(1).name
            # elif float(data) <= float(rows_eucast["R >"].iloc[0]):
            #     interpretation = EucastInterpretation(2).name
            # else:
            #     interpretation = EucastInterpretation(3).name

        df.at[index, antibiotic_name_vitek] = interpretation
        index += 1
    return df


def interpret_vitek(input_vitek, input_eucast, handle_no_matches):
    """Interpret vitek file"""
    df = input_vitek[["LABORNR", "Organism_Code"]].copy()
    # Clean VITEK data
    input_vitek.columns = input_vitek.columns.str.strip()
    input_vitek = input_vitek.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Process each antibiotic column
    for column_vitek in input_vitek.columns[2:]:
        # Adapt VITEK name to EUCAST

        matching_rows_eucast = input_eucast.loc[
            input_eucast["Name"].str.contains(column_vitek, case=False, na=False)
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
                matching_rows_eucast, column_vitek, 70
            )
            # if no match is above 70% cutoff
            if (
                matching_rows_eucast.empty
                and handle_no_matches
                and not column_vitek in ignore_df["Ignorelist"].tolist()
            ):
                handle_no_match(column_vitek, input_eucast)
        # if no match was found and vitek name isnt in ignorelist
        elif not removed_match and not column_vitek in ignore_df["Ignorelist"].tolist():
            print(f"No EUCAST Match: {column_vitek}")
            if handle_no_matches:
                handle_no_match(column_vitek, input_eucast)
        if matching_rows_eucast.empty:
            continue

        # Interpret Match
        column_data_vitek = input_vitek[column_vitek]
        df = get_mic_interpretation(
            column_data_vitek, matching_rows_eucast, column_vitek, df
        )

    return df


# Paths
INPUT_VITEK_PATH = "output/vitek_parsed/"
NAMES_PATH = "resources/names.csv"
IGNORE_PATH = "resources/ignore.csv"
TRANSLATIONS_PATH = "resources/translations.csv"
OUTPUT_FOLDER = "output/interpreted/"

# create missing output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load
names_df = pd.read_csv(NAMES_PATH)
ignore_df = pd.read_csv(IGNORE_PATH)
translations_df = pd.read_csv(TRANSLATIONS_PATH)

# Toggle if no eucast matches should be handled
HANDLE_NO_MATCHES = True

# Interpret & Save
for vitek_file_name in os.listdir(INPUT_VITEK_PATH):
    if vitek_file_name.endswith(".csv"):
        vitek_df = pd.read_csv(INPUT_VITEK_PATH + vitek_file_name)
        eucast_file_name = names_df.loc[
            names_df["Code"] == vitek_df["Organism_Code"].iloc[0], "Eucast_File_Name"
        ].values[0]
        eucast_df = pd.read_json(f"resources/{eucast_file_name}")
        output_df = interpret_vitek(vitek_df, eucast_df, HANDLE_NO_MATCHES)
        output_df.to_csv(OUTPUT_FOLDER + vitek_file_name, index=False)
        print(f"Cleaned file saved to: {OUTPUT_FOLDER + vitek_file_name}")

ignore_df.to_csv(IGNORE_PATH, index=False)
translations_df.to_csv(TRANSLATIONS_PATH, index=False)
