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


def extract_gene_ids(gff_df):
    gff_df["Gene_ID"] = (
        gff_df["attributes"]
        .apply(lambda attr: extract_gene_attribute(attr, "ID"))
        .dropna()
    )

    grouped = gff_df.groupby(ID_COLUMN)["Gene_ID"].apply(list).reset_index()

    return grouped.rename(columns={"Gene_ID": "Resistance Genes"})


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


def encode_one_hot(genotype_df):
    all_genes = set(
        gene for gene_list in genotype_df["Resistance Genes"] for gene in gene_list
    )

    for gene in all_genes:
        genotype_df[gene] = genotype_df["Resistance Genes"].apply(
            lambda genes, g=gene: int(g in genes)
        )

    return genotype_df.drop(columns=["Resistance Genes"])


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
        raw_genotype = extract_gene_ids(raw_gff_df)
        input_genotype = encode_one_hot(raw_genotype).copy()

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
