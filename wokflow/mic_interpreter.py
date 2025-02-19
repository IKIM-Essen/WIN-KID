# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from enum import Enum
import pandas as pd
from fuzzywuzzy import fuzz
import os

OUTPUT_PATH = "output/mic_interpretation.csv"
INPUT_EUCAST_PATH = "resources/MIC_Enterobacterales_v15_0.json"
INPUT_VITEK_PATH = "output/vitek_parsed/"
NAMES_PATH = "resources/names.csv"

class EucastInterpretation(Enum):
    S = 1
    I = 2
    R = 3


def get_most_similar_name(input_df, target_name, cut_off):
    input_df = input_df.copy()
    input_df['Name'] = input_df['Name'].str.split("(", n=1).str[0].str.strip()
    input_df["similarity_score"] = input_df["Name"].apply(
        lambda name: fuzz.ratio(target_name, name)
    )
    best_match_index = input_df["similarity_score"].idxmax()
    best_match = input_df.loc[[best_match_index]].copy()
    if best_match["similarity_score"].iloc[0] < cut_off:
        print(f"No EUCAST Match with Score >= {cut_off}: {target_name}")
        return pd.DataFrame()

    return best_match

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_mic_interpretation(columns_vitek, rows_eucast, antibiotic_name_vitek, df):
    index = 0
    for data in columns_vitek:
        interpretation = ""
        if isinstance(data, float):
            interpretation = "NA"
            print(f"Data not usable: {data}")
        elif is_float(rows_eucast["S <="].iloc[0]) and is_float(rows_eucast["R >"].iloc[0]):
            data = data.replace(">", "").replace("=", "").replace(",", ".")
            if "<" in data:
                interpretation = EucastInterpretation(1).name
            elif float(data) <= float(rows_eucast["S <="].iloc[0]):
                interpretation = EucastInterpretation(1).name
            elif float(data) <= float(rows_eucast["R >"].iloc[0]):
                interpretation = EucastInterpretation(2).name
            else:
                interpretation = EucastInterpretation(3).name
        else:
            interpretation = "NA"
            print("Eucast not usable")

        df.at[index, antibiotic_name_vitek] = interpretation
        index += 1
    return df

def interpret_vitek(input_vitek, input_eucast):
    output_df = input_vitek[["LABORNR", "Organism_Code"]].copy() # Karte rausgenommen & Sample_id_ifh zu LABORNR geändert
    # Clean VITEK data
    input_vitek.columns = input_vitek.columns.str.strip()
    input_vitek = input_vitek.map(lambda x: x.strip() if isinstance(x, str) else x)
    #input_vitek = input_vitek.replace("NA", None)
    #input_vitek = input_vitek.dropna(axis=1, how="all")

    # Process each antibiotic column
    for column_vitek in input_vitek.columns[2:]:
        # Adapt VITEK name to EUCAST
        #column_name_vitek = column_vitek.split("-", 1)[1]
        column_name_vitek = column_vitek.split("(", 1)[0].replace("/", "-").replace("+", "-").replace(" ", "")
        
        matching_rows_eucast = input_eucast.loc[
            input_eucast["Name"].str.contains(column_name_vitek, case=False, na=False)
        ]
        
        if not matching_rows_eucast.empty:
            matching_rows_eucast = get_most_similar_name(
                matching_rows_eucast, column_name_vitek, 70
            )
        else:
            print(f"No EUCAST Match: {column_name_vitek}")
        if matching_rows_eucast.empty:
            continue

        column_data_vitek = input_vitek[column_vitek]
        output_df = get_mic_interpretation(
            column_data_vitek, matching_rows_eucast, column_name_vitek, output_df
        )
    
    return output_df

# Folder durchgehen & alles interpretieren

names_df = pd.read_csv(NAMES_PATH)

for vitek_file_name in os.listdir(INPUT_VITEK_PATH):
    if vitek_file_name.endswith('.csv'):
        input_vitek = pd.read_csv(INPUT_VITEK_PATH + vitek_file_name)
        eucast_file_name = names_df.loc[names_df['Code'] == input_vitek['Organism_Code'].iloc[0], 'Eucast_File_Name'].values[0]
        input_eucast = pd.read_json(f"resources/{eucast_file_name}")
        print(vitek_file_name, eucast_file_name)
        output_df = interpret_vitek(input_vitek, input_eucast)
        # Alle leeren Spalten droppen
        output_df = output_df.replace("NA", pd.NA)
        output_df = output_df.dropna(axis=1, how='all')
        output_df = output_df.fillna("NA")
        output_df.to_csv(f"output/interpreted/{vitek_file_name}", index=False)
        print(f"Cleaned file saved to: output/interpreted/{vitek_file_name}")
