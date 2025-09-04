# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import re
from dataclasses import dataclass, field
from warnings import simplefilter
from itertools import combinations
import pandas as pd
from fuzzywuzzy import fuzz
from constants import GFF_COLUMNS
from constants import RESISTANCE_MAPPING
from constants import ID_COLUMN
from constants import ORGANISM_COLUMN
from constants import MODEL_FOLDER
from config import EXECUTION_MODE
from execution_modes import ExecutionMode


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

GFF_DIR = "resources/genotype"
MIN_SAMPLES_PER_ORG = 70


def extract_gene_attribute(attribute_string, key):
    pattern = rf"{key}=([^;]+)"
    match = re.search(pattern, attribute_string)
    return match.group(1) if match else None


def extract_single_features(gff_df_input, features):
    print(gff_df_input.columns)

    gff_df = gff_df_input.copy()
    for feature in features:
        gff_df[feature] = (
            gff_df["Attributes"]
            .apply(lambda attr, feature=feature: extract_gene_attribute(attr, feature))
            .dropna()
        )
    grouped_features = gff_df.groupby(ID_COLUMN, as_index=False)[features].agg(list)
    gff_df = gff_df.drop_duplicates(subset=[ID_COLUMN])

    grouped_df = pd.merge(
        gff_df.drop(columns=features),
        grouped_features,
        on=ID_COLUMN,
        how="left",
    )
    encoded_df = encode_one_hot(grouped_df, features)

    return encoded_df


def extract_list_features(gff_df_input, features):
    gff_df = gff_df_input.copy()
    for feature in features:
        gff_df[feature] = gff_df["Attributes"].apply(
            lambda attr, feature=feature: (
                extract_gene_attribute(attr, feature) if pd.notna(attr) else None
            )
        )

    grouped_features = gff_df.groupby(ID_COLUMN, as_index=False)[features].agg(
        lambda x: list(filter(pd.notna, x))
    )

    gff_df = gff_df.drop_duplicates(subset=[ID_COLUMN])

    grouped_df = pd.merge(
        gff_df.drop(columns=features),
        grouped_features,
        on=ID_COLUMN,
        how="left",
    )
    for feature in features:
        grouped_df = one_hot_encode_list(grouped_df, feature)

    return grouped_df


def clean_and_split(value):
    if pd.isna(value):
        return []

    cleaned_values = value.strip("[]").replace("'", "").replace('"', "").split(",")

    return [v.strip() for v in cleaned_values if v.strip()]


def load_genotypes(directory_row):
    data = []
    directory = directory_row["PathToGff"]

    for gff_file in os.listdir(directory):
        if gff_file.endswith(".gff"):
            bacterium_id = gff_file.replace(".gff", "")
            gff_path = os.path.join(directory, gff_file)
            gff_df = pd.read_csv(
                gff_path, sep="\t", comment="#", names=GFF_COLUMNS, dtype=str
            )
            gff_df[ID_COLUMN] = bacterium_id

            data.append(gff_df)

    print(
        str(len(data)) + " input genotype samples from " + directory_row["DataSetName"]
    )
    genotype_df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    return genotype_df


def encode_one_hot(df, features):
    for feature in features:
        all_genes = set(gene for gene_list in df[feature] for gene in gene_list)

        new_cols = {
            gene: df[feature].apply(lambda genes, g=gene: int(g in genes))
            for gene in all_genes
        }
        gene_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, gene_df], axis=1)

        df = df.drop(columns=[feature])

    return df


def one_hot_encode_list(df, feature):
    df[feature] = df[feature].astype(str).apply(clean_and_split)

    all_values = set(value for values in df[feature] for value in values)

    new_cols = {
        value: df[feature].apply(lambda x, v=value: int(v in x)) for value in all_values
    }
    value_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, value_df], axis=1)

    df.drop(columns=[feature], inplace=True)

    return df


def find_similar_columns(df, threshold=90):
    columns = df.columns
    for col1, col2 in combinations(columns, 2):
        score = fuzz.ratio(col1, col2)
        if score >= threshold:
            print(f"⚠️ Similar columns: '{col1}' ↔ '{col2}' (Score: {score})")


class DataLoader:
    def __init__(self):
        self.merged_input = None

    def preprocess_phenotype_data(self, dataset_list):
        seen_once = set()
        duplicates = set()
        input_phenotype_list = []
        for _, phenotype_file_row in dataset_list.iterrows():
            input_phenotype_data = pd.read_csv(phenotype_file_row["PathToCsv"])
            print(
                str(len(input_phenotype_data))
                + " input phenotype samples from "
                + phenotype_file_row["DataSetName"]
            )

            # Check for duplicates across datasets
            ids_in_current = set(input_phenotype_data[ID_COLUMN])
            common_ids = ids_in_current & seen_once
            if common_ids:
                duplicates.update(common_ids)
            seen_once.update(ids_in_current)
            input_phenotype_data.columns = list(input_phenotype_data.columns[:2]) + [
                col.capitalize() for col in input_phenotype_data.columns[2:]
            ]
            input_phenotype_list.append(input_phenotype_data)

        if duplicates:
            raise ValueError(
                f"Duplicate IDs found across datasets: {sorted(duplicates)}"
            )
        input_phenotype = pd.concat(input_phenotype_list, ignore_index=True)
        find_similar_columns(input_phenotype)
        print(str(len(input_phenotype)) + " input phenotype samples overall")

        while True:
            rows_to_remove = set()
            updated = False

            for col in input_phenotype.columns[2:]:
                class_counts = input_phenotype[col].value_counts()
                rare_classes = class_counts[class_counts == 1].index

                if not rare_classes.empty:
                    updated = True

                rows_to_remove.update(
                    input_phenotype[input_phenotype[col].isin(rare_classes)].index
                )

            if not updated or EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
                break

            if rows_to_remove:
                print("🔻 Removing rows due to rare classes:")
                print(input_phenotype.loc[sorted(rows_to_remove)])
            input_phenotype = input_phenotype.drop(index=rows_to_remove)

        input_phenotype = input_phenotype.loc[
            :,
            [ID_COLUMN, ORGANISM_COLUMN]
            + list(
                input_phenotype.columns[2:][input_phenotype.iloc[:, 2:].nunique() > 1]
            ),
        ].reset_index(drop=True)

        input_phenotype.fillna(next(iter(RESISTANCE_MAPPING)), inplace=True)
        return input_phenotype

    def preprocess_genotype_data(self, dataset_list):
        raw_gff_list = []
        for _, genotype_file_row in dataset_list.iterrows():
            raw_gff_data = load_genotypes(genotype_file_row)
            raw_gff_list.append(raw_gff_data)
        raw_gff_df = pd.concat(raw_gff_list, ignore_index=True)

        attribute_single_features = ["Name", "ORF"]
        extracted_single_pd = extract_single_features(
            raw_gff_df, attribute_single_features
        )
        extracted_single_pd.drop(columns=GFF_COLUMNS, inplace=True)

        attribute_list_features = ["Antibiotic", "DrugClass", "AMRGeneFamily"]
        extracted_list_pd = extract_list_features(raw_gff_df, attribute_list_features)
        extracted_list_pd.drop(columns=GFF_COLUMNS, inplace=True)

        input_genotype_combined = pd.merge(
            extracted_single_pd,
            extracted_list_pd,
            on=ID_COLUMN,
            how="inner",
        )
        print(str(len(input_genotype_combined)) + " input genotype samples overall")

        return input_genotype_combined

    def get_genotype_data_for_prediction(self, dataset_list):
        input_genotype = self.preprocess_genotype_data(dataset_list)

        # Load allowed features
        features = pd.read_csv("resources/settings/FeatureList.csv", header=None)
        all_features = features[0].tolist()

        # Always keep Sample_ID_IfH
        if "Sample_ID_IfH" not in all_features:
            all_features = ["Sample_ID_IfH"] + all_features

        # Reindex the dataframe: add missing columns (fill with 0), drop extra ones (except Sample_ID_IfH)
        input_genotype = input_genotype.reindex(columns=all_features, fill_value=0)

        # Drop organism column
        input_genotype = input_genotype.drop(ORGANISM_COLUMN, axis=1)

        input_phenotype = self.preprocess_phenotype_data(dataset_list)
        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].astype(str).str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].astype(str).str.strip()

        # Add antibiotic names as columns
        names = [
            os.path.splitext(f)[0]
            for f in os.listdir((MODEL_FOLDER + "second_layer"))
            if f.endswith(".pkl")
        ]
        for col in names:
            if col not in input_phenotype.columns:
                input_phenotype[col] = 0

        # Encode Organism Code
        organism_cat = input_phenotype[ORGANISM_COLUMN].astype("category")
        input_phenotype[ORGANISM_COLUMN] = organism_cat.cat.codes
        organism_mapping = dict(enumerate(organism_cat.cat.categories))

        self.merged_input = pd.merge(
            input_phenotype, input_genotype, on=ID_COLUMN, how="inner"
        )
        num_phenotype_cols = input_phenotype.shape[1]

        feature_cols_merged = list(self.merged_input.columns[num_phenotype_cols:])
        feature_cols_merged.append(ORGANISM_COLUMN)
        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[2:num_phenotype_cols],
            feature_cols_merged,
            organism_mapping=organism_mapping,
        )
        print("Total number of samples: " + str(len(preprocessed_data.merged_input)))
        print("Number of features: " + str(len(preprocessed_data.feature_cols)))
        return preprocessed_data

    def get_preprocessed_data(self, dataset_list):
        input_phenotype = self.preprocess_phenotype_data(dataset_list)
        input_genotype = self.preprocess_genotype_data(dataset_list)

        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].astype(str).str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].astype(str).str.strip()

        # Remove rare species, encode Organism Code as ints and save mapping
        organism_cat = input_phenotype[ORGANISM_COLUMN].astype("category")
        counts = input_phenotype[ORGANISM_COLUMN].value_counts()
        input_phenotype = input_phenotype[
            input_phenotype[ORGANISM_COLUMN].isin(
                counts[counts >= MIN_SAMPLES_PER_ORG].index
            )
        ]
        removed_species = counts[counts < MIN_SAMPLES_PER_ORG]
        print("Removed species (less than " + str(MIN_SAMPLES_PER_ORG) + " samples):")
        print(removed_species)
        input_phenotype[ORGANISM_COLUMN] = organism_cat.cat.codes
        organism_mapping = dict(enumerate(organism_cat.cat.categories))

        # Encode target values as specific ints
        for col in input_phenotype.columns[2:]:
            input_phenotype[col] = (
                input_phenotype[col].map(RESISTANCE_MAPPING).fillna(-1).astype(int)
            )

        # Make target columns distinguishable from feature
        input_phenotype.columns = list(input_phenotype.columns[:2]) + [
            f"{col}_AB" for col in input_phenotype.columns[2:]
        ]
        self.merged_input = pd.merge(
            input_phenotype, input_genotype, on=ID_COLUMN, how="inner"
        )
        num_phenotype_cols = input_phenotype.shape[1]

        feature_cols_merged = list(self.merged_input.columns[num_phenotype_cols:])
        feature_cols_merged.append(ORGANISM_COLUMN)

        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[2:num_phenotype_cols],
            feature_cols_merged,
            organism_mapping=organism_mapping,
        )

        if EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
            pd.DataFrame(preprocessed_data.feature_cols).to_csv(
                "resources/settings/FeatureList.csv", index=False, header=False
            )
        print("Total number of samples: " + str(len(preprocessed_data.merged_input)))
        print("Number of features: " + str(len(preprocessed_data.feature_cols)))
        return preprocessed_data


@dataclass
class PreprocessedDataDTO:

    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
    organism_mapping: dict = field(default_factory=dict)
