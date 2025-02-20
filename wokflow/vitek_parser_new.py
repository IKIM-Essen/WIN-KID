import pandas as pd
from fuzzywuzzy import fuzz
import re
from pprint import pprint
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^a-zA-Z\s]", "", text).lower()

def confirm(question):
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer == "y":
            return True
        elif answer == "n":
            return False
        else:
            print("Invalid Input")

# clean input dataframe
def clean_dataframe(input_df, bacteria, code):
    print(f"Keys used: {bacteria}")
    input_df = input_df[input_df['ERREGERLANG'].isin(bacteria)]
    df = pd.DataFrame(columns=['LABORNR'])

    for row in input_df.itertuples(index=False):
        if row.LABORNR in df['LABORNR'].tolist():
            index = df.loc[df['LABORNR'] == row.LABORNR].index[0]
        else:
            index = len(df)
            df.at[index, 'LABORNR'] = row.LABORNR
        value = (str(row._13) + str(row._14)).replace("nan", "") # _13 = 13th column, _14 = 14th column
        if not value == "":
            df.at[index, row.ANTIBIOTIKA] = value
    
    df.insert(loc=1, column='Organism_Code', value=code)
    df = df.dropna(axis=0, how='all', subset=df.columns[2:])
    df = df.dropna(axis=1, how='all')
    return df

# assign unique bacteria to categories
def assign_unique(df, categories):
    unique_bacteria = df['ERREGERLANG'].dropna().unique()
    assignments = {c: [] for c in categories}

    for b in unique_bacteria:
        scores = []
        for c in categories:
            similarity = fuzz.token_set_ratio(clean_text(b), clean_text(c))
            scores.append(similarity)
        highest_index = scores.index(max(scores))
        assignments[categories[highest_index]].append(b)

    pprint(assignments)
    if confirm("Continue with assignments?"):
        return assignments
    else:
        return []

# combine outputs (optional)
def combine(input_folder):
    combined_df = pd.DataFrame()

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

# fix typos / different names
def translate(input_df, translations_df):
    for old in translations_df['Old']:
        input_df = input_df.rename(columns=lambda col: col.replace(old, translations_df.loc[translations_df['Old'] == old, 'New'].values[0]))
    return input_df

# Paths
INPUT_PATH = "resources/UKM_vitek_daten_2.csv"
NAMES_PATH = "resources/names.csv"
TRANSLATIONS_PATH = "resources/translations.csv"
OUTPUT_FOLDER = "output/vitek_parsed/"

# Load
input_df = pd.read_csv(INPUT_PATH, sep=",")
names_df = pd.read_csv(NAMES_PATH, sep=",")
translations_df = pd.read_csv(TRANSLATIONS_PATH, sep=",")

# Assign all Bacteria to categories
assignments = assign_unique(input_df, names_df['Vitek_Name'])

# Clean & Save (if assignments are correct)
if assignments:
    for name in names_df['Vitek_Name']:
        cleaned_df = clean_dataframe(input_df, assignments[name], names_df.loc[names_df['Vitek_Name'] == name, 'Code'].values[0])
        cleaned_df = translate(cleaned_df, translations_df)
        cleaned_df.to_csv(OUTPUT_FOLDER + f"{name.lower().replace(" ", "_")}.csv", index=False)
        print(f"All {name} saved to {OUTPUT_FOLDER + f"{name.lower().replace(" ", "_")}.csv"}")
