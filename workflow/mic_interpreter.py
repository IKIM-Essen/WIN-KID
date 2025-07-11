# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import argparse
import os
from enum import Enum
import pandas as pd
from fuzzywuzzy import fuzz
from constants import NAMES_PATH
from constants import TRANSLATIONS_PATH
from constants import IGNORE_PATH
from constants import INPUT_EUCAST_FOLDER


# Load
NAMES_DF = pd.read_csv(NAMES_PATH)
IGNORE_DF = pd.read_csv(IGNORE_PATH)
TRANSLATION_DF = pd.read_csv(TRANSLATIONS_PATH)


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
        if not target_name in IGNORE_DF["Ignorelist"].tolist():
            print(f"No EUCAST Match with Score >= {cut_off}: {target_name}")
        return pd.DataFrame()

    return best_match


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

        df = df.reset_index(drop=True)
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
        if matching_rows_eucast.empty:
            print(f"No EUCAST Match: {column_vitek}")

        for index, row in matching_rows_eucast.iterrows():
            if (not is_float(row["S <="])) or (not is_float(row["R >"])):
                matching_rows_eucast = matching_rows_eucast.drop(index)

        if not matching_rows_eucast.empty:
            matching_rows_eucast = get_most_similar_name(
                matching_rows_eucast, column_vitek, 70
            )
        if matching_rows_eucast.empty:
            continue
        column_data_vitek = input_vitek[column_vitek]
        df = get_mic_interpretation(
            column_data_vitek, matching_rows_eucast, column_vitek, df
        )
    # Convert all data types to object
    df = df.astype("object")
    return df


def interpret_folder(vitek_folder, output_folder_df):
    output_df_dic = {}
    for vitek_file_name in os.listdir(vitek_folder):
        if vitek_file_name.endswith(".csv"):
            vitek_path = os.path.join(vitek_folder, vitek_file_name)
            output_path = os.path.join(
                output_folder_df, vitek_file_name.replace("parsed", "interpreted")
            )

            vitek_df = pd.read_csv(vitek_path)
            split_df = {key: group for key, group in vitek_df.groupby("Organism_Code")}
            output_df = pd.DataFrame()

            for key, df in split_df.items():
                matching_json_name = NAMES_DF.loc[
                    NAMES_DF["Code"] == key,
                    "Eucast_File_Name",
                ].values[0]
                print(f"Used {matching_json_name} for {os.path.basename(vitek_path)}")
                matching_json = pd.read_json(INPUT_EUCAST_FOLDER + matching_json_name)
                interpreted_df = interpret_vitek(df, matching_json)
                output_df = pd.concat(
                    [output_df, interpreted_df], ignore_index=True
                ).fillna("NA")
            output_df_dic[output_path] = output_df
    return output_df_dic


# terminal input
if __name__ == "__main__":

    # LOAD
    parser = argparse.ArgumentParser(
        description="Parse Vitek data and categorize bacteria."
    )
    parser.add_argument(
        "input_folder_path", help="Path to input folder containing CSV files"
    )
    parser.add_argument(
        "output_folder_path", help="Path to output directory for interpreted files"
    )
    args = parser.parse_args()

    input_folder = args.input_folder_path
    output_folder = args.output_folder_path

    # create missing output directory
    os.makedirs(output_folder, exist_ok=True)

    # PROCESS
    # TODO: Enables the merging of columns that mean the same substance. Example: Fosfomycin and fosfomycin iv in the UME data.
    outputs = interpret_folder(input_folder, output_folder)

    # SAVE
    for path, output in outputs.items():
        output.to_csv(path, index=False)
        print(f"Interpreted file saved to: {path}")

    IGNORE_DF.to_csv(IGNORE_PATH, index=False)
    TRANSLATION_DF.to_csv(TRANSLATIONS_PATH, index=False)
