import pandas as pd


def clean_dataframe(df):
    df = df.loc[:, ~df.columns.str.contains("Family|Andere-Instrument", regex=True)]
    df.columns = df.columns.str.strip()
    columns_to_drop = [
        "Lab_ID",
        "Isolate_Number",
        "Patient_ID",
        "Specimen_Source",
        "Collection_Date.x",
        "Testing_Date",
        "Organism_Name",
        "Card_Type",
        "Lot_Number",
        "Expiration_Date",
        "Bar_Code",
    ]
    return df.drop(columns=columns_to_drop, errors="ignore")


# Paths
INPUT_PATH = "resources/MHK_UKM_Subset.csv"
OUTPUT_PATH = "output/vitek_parsed.csv"

# Load
input_df = pd.read_csv(INPUT_PATH, sep="\t")

# Clean
cleaned_df = clean_dataframe(input_df)

# Save
cleaned_df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned file saved to: {OUTPUT_PATH}")
