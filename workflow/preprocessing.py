# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import re
from dataclasses import dataclass
import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from constants import GFF_COLUMNS


GFF_DIR = "resources/genotype"
ID_COLUMN = "Sample_ID_IfH"
ORGANISM_COLUMN = "Organism_Code"


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


def load_genotypes(directory):
    data = []

    for gff_file in os.listdir(directory):
        if gff_file.endswith(".gff"):
            bacterium_id = gff_file.replace(".gff", "")
            gff_path = os.path.join(directory, gff_file)
            gff_df = pd.read_csv(
                gff_path, sep="\t", comment="#", names=GFF_COLUMNS, dtype=str
            )
            gff_df[ID_COLUMN] = bacterium_id

            data.append(gff_df)

    print("Number of input genotype samples: " + str(len(data)))
    genotype_df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    return genotype_df


def encode_one_hot(df, features):
    for feature in features:
        all_genes = set(gene for gene_list in df[feature] for gene in gene_list)

        for gene in all_genes:
            df[gene] = df[feature].apply(lambda genes, g=gene: int(g in genes))

        df = df.drop(columns=[feature])

    return df


def one_hot_encode_list(df, feature):
    df[feature] = df[feature].astype(str).apply(clean_and_split)

    all_values = set(value for values in df[feature] for value in values)

    for value in all_values:
        df[value] = df[feature].apply(lambda x, value=value: int(value in x))

    df.drop(columns=[feature], inplace=True)

    return df


class DataLoader:
    def __init__(self):
        self.merged_input = None

    def preprocess_phenotype_data(self, phenotype_file_paths):
        seen_once = set()
        duplicates = set()
        input_phenotype_list = []
        for phenotype_file_path in phenotype_file_paths:
            input_phenotype_data = pd.read_csv(phenotype_file_path)

            # Check for duplicates across datasets
            ids_in_current = set(input_phenotype_data[ID_COLUMN])
            common_ids = ids_in_current & seen_once
            if common_ids:
                duplicates.update(common_ids)
            seen_once.update(ids_in_current)

            input_phenotype_list.append(input_phenotype_data)

        if duplicates:
            raise ValueError(f"Duplicate IDs found across datasets: {sorted(duplicates)}")
        input_phenotype = pd.concat(input_phenotype_list, ignore_index=True)
        print("Number of input phenotype samples: " + str(len(input_phenotype)))

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

            if not updated:
                break

            input_phenotype = input_phenotype.drop(index=rows_to_remove)

        input_phenotype = input_phenotype.loc[
            :,
            [ID_COLUMN, ORGANISM_COLUMN]
            + list(
                input_phenotype.columns[2:][input_phenotype.iloc[:, 2:].nunique() > 1]
            ),
        ].reset_index(drop=True)

        input_phenotype.fillna("S", inplace=True)
        return input_phenotype

    def preprocess_genotype_data(self, genotype_dir_paths):
        raw_gff_list = []
        for genotype_dir_path in genotype_dir_paths:
            raw_gff_data = load_genotypes(genotype_dir_path)
            raw_gff_list.append(raw_gff_data)
        raw_gff_df = pd.concat(raw_gff_list, ignore_index=True)
        raw_gff_df.fillna("0", inplace=True)

        attribute_single_features = ["Name", "ResistanceMechanism", "ORF"]
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

        return input_genotype_combined

    def get_preprocessed_data(self, dataset_list):
        input_phenotype = self.preprocess_phenotype_data(dataset_list["PathToCsv"])
        input_genotype = self.preprocess_genotype_data(dataset_list["PathToGff"])

        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].astype(str).str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].astype(str).str.strip()

        # Encode Organism Code as ints
        input_phenotype[ORGANISM_COLUMN] = input_phenotype[ORGANISM_COLUMN].astype("category").cat.codes
        # Encode target values as specific ints
        mapping = {"S": 0, "I": 1, "R": 2}
        for col in input_phenotype.columns[2:]:
            input_phenotype[col] = (
                input_phenotype[col].map(mapping).fillna(-1).astype(int)
            )

        self.merged_input = pd.merge(
            input_phenotype, input_genotype, on=ID_COLUMN, how="inner"
        )

        # for column in self.merged_input.columns:
            # print(column)

        num_phenotype_cols = input_phenotype.shape[1]
        print(input_phenotype.columns)
        
        feature_cols_merged = list(self.merged_input.columns[num_phenotype_cols:])
        feature_cols_merged.append(ORGANISM_COLUMN) 

        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[2:num_phenotype_cols],
            feature_cols_merged,
        )
        print("Number of preprocessed merged samples: " + str(len(preprocessed_data.merged_input)))

        #TODO: Fix AB_x -> There seems to be some mix up with the preprocessed_data.target_cols and the feature_cols_merged
        print(preprocessed_data.target_cols)
        return preprocessed_data


@dataclass
class PreprocessedDataDTO:

    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
