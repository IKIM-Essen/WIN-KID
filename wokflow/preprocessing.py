import os
import re
from dataclasses import dataclass
import pandas as pd

GFF_DIR = "resources/genotype"
GFF_COLUMNS = [
    "seqid",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
]
ID_COLUMN = "Sample_ID_IfH "


def extract_gene_attribute(attribute_string, key):
    pattern = rf"{key}=([^;]+)"
    match = re.search(pattern, attribute_string)
    return match.group(1) if match else None


def extract_single_features(gff_df, features):
    # Extract specified features
    for feature in features:
        gff_df[feature] = (
            gff_df["attributes"]
            .apply(lambda attr: extract_gene_attribute(attr, feature))
            .dropna()
        )
    # Group only the feature columns, keeping "attributes" and all other columns
    grouped_features = gff_df.groupby(ID_COLUMN, as_index=False)[features].agg(list)

    # Drop duplicate rows (but keep all non-feature columns)
    gff_df = gff_df.drop_duplicates(subset=[ID_COLUMN])

    # Merge back to restore all original columns, but avoid '_x' issues
    grouped_df = pd.merge(
        gff_df.drop(columns=features),
        grouped_features,
        on=ID_COLUMN,
        how="left",
    )
    # Apply one-hot encoding while keeping all other columns
    encoded_df = encode_one_hot(grouped_df, features)

    return encoded_df


def extract_list_features(gff_df, features):
    # Extract specified features from attributes
    for feature in features:
        gff_df[feature] = gff_df["attributes"].apply(
            lambda attr: (
                extract_gene_attribute(attr, feature) if pd.notna(attr) else None
            )
        )

    # Group only the feature columns, keeping "attributes" and all other columns intact
    grouped_features = gff_df.groupby(ID_COLUMN, as_index=False)[features].agg(
        lambda x: list(filter(pd.notna, x))
    )

    # Drop duplicates (but keep all non-feature columns)
    gff_df = gff_df.drop_duplicates(subset=[ID_COLUMN])

    # Merge back to restore all original columns while avoiding "_x" issues
    grouped_df = pd.merge(
        gff_df.drop(columns=features),
        grouped_features,
        on=ID_COLUMN,
        how="left",
    )

    # Apply one-hot encoding for each feature column
    for feature in features:
        grouped_df = one_hot_encode_column(grouped_df, feature)

    return grouped_df


def clean_and_split(value):
    if pd.isna(value):  # Handle NaN values
        return []

    # Remove brackets and single quotes, then split on commas
    cleaned_values = value.strip("[]").replace("'", "").replace('"', "").split(",")

    # Strip spaces around each value
    return [v.strip() for v in cleaned_values if v.strip()]


def one_hot_encode_column(df, column):
    # Convert comma-separated values into lists
    df[column] = df[column].astype(str).apply(clean_and_split)

    # Extract all unique values across all rows
    all_values = set(value for values in df[column] for value in values)

    # Create binary columns for each unique value
    for value in all_values:
        df[value] = df[column].apply(lambda x: int(value in x))

    # Drop original column
    df.drop(columns=[column], inplace=True)
    # TODO: Drop?
    # print(df.columns)
    # df.drop(columns=["nan"], inplace=True)

    return df


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


def encode_one_hot(genotype_df, features):
    for feature in features:
        all_genes = set(
            gene for gene_list in genotype_df[feature] for gene in gene_list
        )

        for gene in all_genes:
            genotype_df[gene] = genotype_df[feature].apply(
                lambda genes, g=gene: int(g in genes)
            )

        genotype_df = genotype_df.drop(columns=[feature])

    return genotype_df


class DataLoader:
    def __init__(
        self, phenotype_path="output/mic_interpretation.csv", genotype_dir=GFF_DIR
    ):
        self.phenotype_path = phenotype_path
        self.genotype_dir = genotype_dir
        self.merged_input = self._load_and_merge_data()

    def _load_and_merge_data(self):
        input_phenotype = pd.read_csv(self.phenotype_path)

        raw_gff_df = load_genotypes(self.genotype_dir)

        attribute_single_features = ["Name", "ResistanceMechanism"]
        extracted_single_pd = extract_single_features(
            raw_gff_df, attribute_single_features
        )

        attribute_list_features = ["Antibiotic"]
        extracted_list_pd = extract_list_features(raw_gff_df, attribute_list_features)

        for i in extracted_list_pd.columns:
            print(i)
        extracted_single_pd.drop(columns=GFF_COLUMNS, inplace=True)
        extracted_list_pd.drop(columns=GFF_COLUMNS, inplace=True)
        merged_feature_df = pd.merge(
            extracted_single_pd,
            extracted_list_pd,
            on=ID_COLUMN,
            how="inner",
        )

        input_genotype = merged_feature_df

        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].str.strip()

        return pd.merge(input_phenotype, input_genotype, on=ID_COLUMN, how="inner")

    def get_preprocessed_data(self):
        """Returns preprocessed data as a DTO."""
        return PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[3:17],
            self.merged_input.columns[17:],
        )


@dataclass
class PreprocessedDataDTO:
    """Data Transfer Object for preprocessed input data."""

    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
