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
    input_df["similarity_score"] = input_df["Name"].apply(
        lambda name: fuzz.ratio(target_name, name)
    )
    best_match_index = input_df["similarity_score"].idxmax()
    best_match = input_df.loc[[best_match_index]].copy()
    if best_match["similarity_score"].iloc[0] < cut_off:
        print("No EUCAST Match with Score >= " + str(cut_off) + ": " + column)
        data = []
        best_match = pd.DataFrame(data)

    return best_match


def interpret_mic(columns, rows, df):
    index = 0
    for data in columns:
        interpretation = ""
        data = data.replace(">", "").replace("<", "").replace("=", "").replace(",", ".")
        data = float(data)
        if data <= float(rows["S <="]):
            interpretation = EucastInterpretation(1).name
        elif data <= float(rows["R >"]):
            interpretation = EucastInterpretation(2).name
        else:
            interpretation = EucastInterpretation(3).name

        df.at[index, column_name] = interpretation
        index += 1
    return df


input_vitek = pd.read_csv(INPUT_VITEK_PATH)
input_eucast = pd.read_json(INPUT_EUCAST_PATH)

output_df = input_vitek[["Sample_ID_IfH", "Organism_Code", "Card_Name"]].copy()

# Clean VITEK file
input_vitek.columns = input_vitek.columns.str.strip()
input_vitek = input_vitek.applymap(lambda x: x.strip() if isinstance(x, str) else x)
input_vitek = input_vitek.replace("NA", None)
input_vitek = input_vitek.dropna(axis=1, how="all")


for column in input_vitek.columns[3:]:
    # ToDo: Encapsulate
    column_data = input_vitek[column]
    column_name = column.split("-", 1)[1]  # Split at the first "-" and keep the suffix
    column_name = column_name.replace("/", "-")

    matching_rows = input_eucast.loc[
        input_eucast["Name"].str.contains(column_name, case=False, na=False)
    ]

    if not matching_rows.empty:
        matching_rows = get_most_similar_name(matching_rows, column_name, 70)

    else:
        print("No EUCAST Match: " + column)

    if matching_rows.empty:
        continue

    output_df = interpret_mic(column_data, matching_rows, output_df)

output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned file saved to: {OUTPUT_PATH}")
