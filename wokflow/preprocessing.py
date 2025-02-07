import os
import re
from dataclasses import dataclass
import pandas as pd

# Constants
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


def extract_gene_id(attribute_string):
    match = re.search(r"ID=([^;]+)", attribute_string)
    return match.group(1) if match else None


def extract_gene_names(gff_file_path):
    gff_data = pd.read_csv(
        gff_file_path, sep="\t", comment="#", names=GFF_COLUMNS, dtype=str
    )
    return gff_data["attributes"].apply(extract_gene_id).dropna().tolist()


def load_genotypes(directory):
    bacteria_genes = {
        gff_file.replace(".gff", ""): extract_gene_names(
            os.path.join(directory, gff_file)
        )
        for gff_file in os.listdir(directory)
        if gff_file.endswith(".gff")
    }
    return pd.DataFrame(bacteria_genes.items(), columns=[ID_COLUMN, "Resistance Genes"])


def encode_one_hot(genotype_df):
    all_genes = set(
        gene for gene_list in genotype_df["Resistance Genes"] for gene in gene_list
    )
    for gene in all_genes:
        genotype_df[gene] = genotype_df["Resistance Genes"].apply(
            lambda genes, g=gene: int(g in genes)
        )

    genotype_df = genotype_df.drop(columns=["Resistance Genes"])
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
        raw_genotype = load_genotypes(self.genotype_dir)
        input_genotype = encode_one_hot(raw_genotype).copy()

        # Cleanup
        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].str.strip()

        return pd.merge(input_phenotype, input_genotype, on=ID_COLUMN, how="inner")

    def get_preprocessed_data(self):
        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[3:17],
            self.merged_input.columns[17:],
        )
        return preprocessed_data


@dataclass
class PreprocessedDataDTO:
    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
