from enum import Enum
import pandas as pd
from fuzzywuzzy import fuzz

OUTPUT_PATH = "output/mic_interpretation.csv"
INPUT_EUCAST_PATH = "resources/MIC_Enterobacterales_v15_0.json"
INPUT_VITEK_PATH = "output/vitek_parsed.csv"


class EucastInterpretation(Enum):
    S = 1
    I = 2
    R = 3


def get_most_similar_name(input_df, target_name, cut_off):
    input_df = input_df.copy()
    input_df["similarity_score"] = input_df["Name"].apply(
        lambda name: fuzz.ratio(target_name, name)
    )
    best_match_index = input_df["similarity_score"].idxmax()
    best_match = input_df.loc[[best_match_index]].copy()
    if best_match["similarity_score"].iloc[0] < cut_off:
        print(f"No EUCAST Match with Score >= {cut_off}: {target_name}")
        return pd.DataFrame()

    return best_match


def get_mic_interpretation(columns_vitek, rows_eucast, antibiotic_name_vitek, df):
    index = 0
    for data in columns_vitek:
        interpretation = ""
        data = data.replace(">", "").replace("<", "").replace("=", "").replace(",", ".")
        data = float(data)
        if data <= float(rows_eucast["S <="].iloc[0]):
            interpretation = EucastInterpretation(1).name
        elif data <= float(rows_eucast["R >"].iloc[0]):
            interpretation = EucastInterpretation(2).name
        else:
            interpretation = EucastInterpretation(3).name

        df.at[index, antibiotic_name_vitek] = interpretation
        index += 1
    return df


input_vitek = pd.read_csv(INPUT_VITEK_PATH)
input_eucast = pd.read_json(INPUT_EUCAST_PATH)

output_df = input_vitek[["Sample_ID_IfH", "Organism_Code", "Card_Name"]].copy()

# Clean VITEK data
input_vitek.columns = input_vitek.columns.str.strip()
input_vitek = input_vitek.map(lambda x: x.strip() if isinstance(x, str) else x)
input_vitek = input_vitek.replace("NA", None)
input_vitek = input_vitek.dropna(axis=1, how="all")

# Process each antibiotic column
for column_vitek in input_vitek.columns[3:]:
    # Adapt VITEK name to EUCAST
    column_name_vitek = column_vitek.split("-", 1)[1]
    column_name_vitek = column_name_vitek.replace("/", "-")

    matching_rows_eucast = input_eucast.loc[
        input_eucast["Name"].str.contains(column_name_vitek, case=False, na=False)
    ]

    if not matching_rows_eucast.empty:
        matching_rows_eucast = get_most_similar_name(
            matching_rows_eucast, column_name_vitek, 70
        )
    else:
        print(f"No EUCAST Match: {column_vitek}")
    if matching_rows_eucast.empty:
        continue

    column_data_vitek = input_vitek[column_vitek]
    output_df = get_mic_interpretation(
        column_data_vitek, matching_rows_eucast, column_name_vitek, output_df
    )

output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned file saved to: {OUTPUT_PATH}")
