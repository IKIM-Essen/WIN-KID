# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import re
from dataclasses import dataclass
import pandas as pd
from constants import GFF_COLUMNS

GFF_DIR = "resources/genotype"
ID_COLUMN = "Sample_ID_IfH"


def extract_gene_attribute(attribute_string, key):
    pattern = rf"{key}=([^;]+)"
    match = re.search(pattern, attribute_string)
    return match.group(1) if match else None


def extract_single_features(gff_df_input, features):
    gff_df = gff_df_input.copy()
    for feature in features:
        gff_df[feature] = (
            gff_df["attributes"]
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
        gff_df[feature] = gff_df["attributes"].apply(
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

    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


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

    def get_preprocessed_data(self, phenotype_file_path, genotype_dir_path):
        input_phenotype = pd.read_csv(phenotype_file_path)

        raw_gff_df = load_genotypes(genotype_dir_path)

        # Replace all NaN values with "S" early
        raw_gff_df.fillna("S", inplace=True)

        attribute_single_features = ["Name", "ResistanceMechanism"]
        extracted_single_pd = extract_single_features(
            raw_gff_df, attribute_single_features
        )
        extracted_single_pd.drop(columns=GFF_COLUMNS, inplace=True)

        attribute_list_features = ["Antibiotic"]
        extracted_list_pd = extract_list_features(raw_gff_df, attribute_list_features)
        extracted_list_pd.drop(columns=GFF_COLUMNS, inplace=True)

        input_genotype = pd.merge(
            extracted_single_pd,
            extracted_list_pd,
            on=ID_COLUMN,
            how="inner",
        )

        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].str.strip()

        self.merged_input = pd.merge(
            input_phenotype, input_genotype, on=ID_COLUMN, how="inner"
        )

        # Replace any remaining NaNs in the entire dataset
        self.merged_input = pd.merge(
            input_phenotype, input_genotype, on=ID_COLUMN, how="inner"
        ).fillna("S")

        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[2:22],  # adjust to current table
            self.merged_input.columns[22:],  # same
        )

        return preprocessed_data


@dataclass
class PreprocessedDataDTO:

    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
