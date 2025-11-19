# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from dataclasses import dataclass

FASTA_DIR = "/groups/ds/Win-KID/BVBRC/vitek_ii_cleaned/VITEK_cleaned/"
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


@dataclass
class RandomForestSettings:
    n_estimators: int
    class_weight: str
    max_depth: object
    min_samples_split: int
    min_samples_leaf: int
    max_features: str
    bootstrap: bool


CLASSIC_RF_SETTINGS = RandomForestSettings(
    n_estimators=100,
    class_weight="balanced_subsample",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=0.3,
    bootstrap=True,
)

STACKED_RF_SETTINGS = RandomForestSettings(
    n_estimators=1500,
    class_weight="balanced_subsample",
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=0.3,
    bootstrap=True,
)
