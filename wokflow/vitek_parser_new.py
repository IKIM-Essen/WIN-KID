import pandas as pd
from fuzzywuzzy import fuzz
import re

def clean_dataframe(input_df, bacteria):
    bakterien = ""
    # Filter Zeilen mit token_set_ratio damit nach stichwörtern gesucht wird
    input_df = input_df[input_df['ERREGERLANG'].apply(lambda x: fuzz.token_set_ratio(str(x).lower(), bacteria.lower()) >= 50 if pd.notna(x) else False)]
    # Leeres Dataframe nur mit spalte 'LABORNR' erzeugen
    df = pd.DataFrame(columns=['LABORNR'])
    # Alle Rows durchgehen
    for index, row in input_df.iterrows():
        if not row['ERREGERLANG'] in bakterien:
            bakterien += row['ERREGERLANG'] + ","
        # Falls Labornr nicht vorhanden -> neue Zeile erstellen
        if not df['LABORNR'].isin([row['LABORNR']]).any():
            new_row = {col: (row['LABORNR'] if col == 'LABORNR' else 'NA') for col in df.columns}
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index = True)
        # Falls Antibiotika nicht vorhanden -> neue Spalte erstellen
        if not row['ANTIBIOTIKA'] in df.columns:
            df[row['ANTIBIOTIKA']] = 'NA'
        # Wert einsetzen
        i = df.loc[df['LABORNR'] == row['LABORNR']].index[0]
        df.at[i, row['ANTIBIOTIKA']] = (str(row['TESTUNG']) + str(row['MHK-VKZ']) + str(row['MHK-Wert'])).replace("nan", "")
    print(bakterien)
    return df

# Allgemeine Ähnlichkeit der einzigartigen Bakterienbekennzeichnungen überprüfen
def check_similarity(df, bacteria):
    unique_bacteria = df['ERREGERLANG'].dropna().unique()

    print(f"🔍 Überprüfung der Ähnlichkeit mit: '{bacteria}'")
    for b in unique_bacteria:
        similarity = fuzz.token_set_ratio(clean_text(b), clean_text(bacteria))
        print(f"{b} : {similarity}")

# Sonderzeichen Entfernen
def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^a-zA-Z\s]", "", text).lower()


# Paths
INPUT_PATH = "resources/UKM_vitek_daten_2.csv"
OUTPUT_PATH = "output/vitek_parsed_ecoli.csv"

# Load
input_df = pd.read_csv(INPUT_PATH, sep=",")

# Clean
cleaned_df = clean_dataframe(input_df, "Escherichia coli")

# Save
cleaned_df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned file saved to: {OUTPUT_PATH}")