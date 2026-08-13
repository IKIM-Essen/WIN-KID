# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from dataclasses import dataclass

NAMES_PATH = "resources/settings/names.csv"
TRANSLATIONS_PATH = "resources/settings/translations.csv"
IGNORE_PATH = "resources/settings/ignore.csv"
INPUT_EUCAST_FOLDER = "resources/eucast_files/"
MODEL_FOLDER = "rf_models/"
W2V_MODEL_PATH = "w2v_models/w2v_model.bin"
PCA_MODEL_PATH = "pca_models/pca_model.joblib"


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


@dataclass
class w2vSettings:
    k_size: int
    vec_size: int
    w2v_epochs: int
    include_position: bool
    train_subset_size: int
    window: int
    batch_size: int
    min_count: int
    sg: int
    negative: int
    tuning_trials: int
    sample: float
    seed: int


W2V_SETTINGS = w2vSettings(
    k_size=8,
    vec_size=50,
    w2v_epochs=3,
    include_position=False,
    train_subset_size=1000,
    window=15,
    batch_size=300,
    min_count=8,
    sg=0,
    negative=5,
    tuning_trials=10,
    sample=1e-4,
    seed=42,
)
