# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

NAMES_PATH = "resources/settings/names.csv"
TRANSLATIONS_PATH = "resources/settings/translations.csv"
IGNORE_PATH = "resources/settings/ignore.csv"
INPUT_EUCAST_FOLDER = "resources/eucast_files/"
MODEL_FOLDER = "rf_models/"

GFF_COLUMNS = [
    "SeqID",
    "Source",
    "Type",
    "Start",
    "End",
    "Score",
    "Strand",
    "Phase",
    "Attributes",
]
ID_COLUMN = "Sample_ID_IfH"
ORGANISM_COLUMN = "Organism_Code"

RESISTANCE_MAPPING = {"NA": 0, "S": 1, "I": 2, "R": 3}
