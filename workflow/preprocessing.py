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
from Bio import SeqIO
import gzip
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import random
import logging

logging.basicConfig(
    filename="preprocessing_log.txt",
    filemode="w",  # use 'a' to append if needed
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


K = 6
VEC_SIZE = 40
W2V_EPOCHS = 5
FASTA_DIR = "/groups/ds/Win-KID/BVBRC/vitek_ii_cleaned/VITEK_cleaned"
INCLUDE_POSITION = False
random.seed(42)

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

GFF_DIR = "resources/genotype"


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


def find_similar_columns(df, threshold=90):
    columns = df.columns
    for col1, col2 in combinations(columns, 2):
        score = fuzz.ratio(col1, col2)
        if score >= threshold:
            print(f"⚠️ Similar columns: '{col1}' ↔ '{col2}' (Score: {score})")


def fasta_to_kmers(fasta_path, k=K):
    kmers = []
    open_func = gzip.open if fasta_path.endswith(".gz") else open
    with open_func(fasta_path, "rt") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        for record in records:
            seq = str(record.seq).upper()
            kmers.extend([
                seq[i:i+k] for i in range(len(seq)-k+1)
                if set(seq[i:i+k]).issubset({'A', 'C', 'G', 'T'})
            ])
    return kmers

# --- New generator for streaming ---
def iter_kmer_sequences(fasta_ids, fasta_dir, k=K):
    """
    Generator that yields one k-mer sequence list at a time.
    Does NOT store all sequences in memory at once.
    """
    for sid in fasta_ids:
        fpath = os.path.join(fasta_dir, f"{sid}.fna.gz")
        if os.path.exists(fpath):
            yield fasta_to_kmers(fpath, k)


# --- Updated train_word2vec_model with streaming & batching ---
def train_word2vec_model_streaming(all_fasta_ids, fasta_dir, vector_size=VEC_SIZE, window=5, min_count=2):
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))  # Use all allocated CPUs
    
    # 1. Create empty model
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sample=1e-4,    # subsample frequent k-mers
        negative=5,
        epochs=W2V_EPOCHS
    )
    
    # 2. Build vocab from all sequences (streaming)
    print("📦 Building vocabulary...")
    model.build_vocab(iter_kmer_sequences(all_fasta_ids, fasta_dir, k=K))
    print(f"✅ Vocabulary size: {len(model.wv)} k-mers")

    # 3. Train in batches
    batch_size = 300  # Number of FASTA files per batch
    for i in range(0, len(all_fasta_ids), batch_size):
        batch_ids = all_fasta_ids[i:i+batch_size]
        print(f"🚀 Training batch {i//batch_size+1} on {len(batch_ids)} files...")
        model.train(
            iter_kmer_sequences(batch_ids, fasta_dir, k=K),
            total_examples=len(batch_ids),
            epochs=epochs
        )

    return model

def encode_sample(sample_id, fasta_dir, model, k=K):
    fpath = os.path.join(fasta_dir, f"{sample_id}.fna.gz")
    if not os.path.exists(fpath):
        return None
    emb_vectors = []
    open_func = gzip.open if fpath.endswith(".gz") else open
    with open_func(fpath, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq).upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if set(kmer).issubset({'A', 'C', 'G', 'T'}) and kmer in model.wv:
                    vec = model.wv[kmer]
                    if INCLUDE_POSITION:
                        rel_pos = i / len(seq)
                        vec = np.append(vec, rel_pos)
                    emb_vectors.append(vec)
    if not emb_vectors:
        return None
    return sample_id, np.mean(emb_vectors, axis=0)

def encode_all_samples(fasta_ids, fasta_dir, model, k=K):
    all_vecs = []
    all_ids = []
    for sid in fasta_ids:
        result = encode_sample(sid, fasta_dir, model, k)
        if result:
            sample_id, mean_vec = result
            all_ids.append(sample_id)
            all_vecs.append(mean_vec)
    if not all_vecs:
        raise ValueError("No k-mer embeddings generated!")
    cols = [f'kmer_{i}' for i in range(len(all_vecs[0]))]
    df = pd.DataFrame(all_vecs, columns=cols)
    df[ID_COLUMN] = all_ids
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df



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

    def get_preprocessed_data(self, dataset_list):
        input_phenotype = self.preprocess_phenotype_data(dataset_list)
        input_genotype = self.preprocess_genotype_data(dataset_list)

        logger.info(f"📊 Initial input_phenotype shape: {input_phenotype.shape}")
        logger.info(f"📊 Initial input_genotype shape: {input_genotype.shape}")

        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].astype(str).str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].astype(str).str.strip()

        # Encode Organism Code as ints and save mapping
        organism_cat = input_phenotype[ORGANISM_COLUMN].astype("category")
        input_phenotype[ORGANISM_COLUMN] = organism_cat.cat.codes
        organism_mapping = dict(enumerate(organism_cat.cat.categories))

        # Encode target values as specific ints
        for col in input_phenotype.columns[2:]:
            input_phenotype[col] = (
                input_phenotype[col].map(RESISTANCE_MAPPING).fillna(0).astype(int)
            )

        logger.info("✅ After encoding target values:")
        logger.info(input_phenotype.dtypes)
        logger.info(input_phenotype.head(4))

        # Make target columns distinguishable from feature
        input_phenotype.columns = list(input_phenotype.columns[:2]) + [
            f"{col}_AB" for col in input_phenotype.columns[2:]
        ]
        self.merged_input = pd.merge(
            input_phenotype, input_genotype, on=ID_COLUMN, how="inner"
        )
        num_phenotype_cols = input_phenotype.shape[1]

         # 🔍 Print merged structure
        logger.info("\n📊 Merged Data Summary (Phenotype + Genotype):")
        logger.info(f"🔢 Shape: {self.merged_input.shape}")
        logger.info(f"🧬 Columns: {self.merged_input.columns[:5].tolist()} ...")
        logger.info(f"🧪 Column Types:\n{self.merged_input.dtypes.head()}")
        logger.info(f"🧾 Sample row:\n{self.merged_input.head(1)}\n")


        feature_cols_merged = list(self.merged_input.columns[num_phenotype_cols:])
        feature_cols_merged.append(ORGANISM_COLUMN)

        # add kmere embaddings
        fasta_dir = FASTA_DIR
        fasta_ids = self.merged_input[ID_COLUMN].unique().tolist()

        TRAIN_SUBSET_SIZE = 1000
        train_ids = random.sample(fasta_ids, min(TRAIN_SUBSET_SIZE, len(fasta_ids)))
        #train_ids = fasta_ids
        logger.info(f"Train Word2Vec on all {len(train_ids)} FASTA files using streaming...")
        w2v_model = train_word2vec_model_streaming(train_ids, fasta_dir, vector_size=VEC_SIZE, window=4, min_count=2)


        logger.info("Generate aggregated k-mer embeddings...")
        embedding_df = encode_all_samples(train_ids, fasta_dir, w2v_model)

        logger.info(f"🧬 Embedding DataFrame shape: {embedding_df.shape}")
        logger.info(f"🧬 Embedding columns: {embedding_df.columns.tolist()[:5]}...")


        logger.info(f"Add {embedding_df.shape[1]-1} k-mer embeddings to feature columns ...")
        self.merged_input = pd.merge(self.merged_input, embedding_df, on=ID_COLUMN, how="left")
        # Entferne alle Zeilen, bei denen keine Kmer-Embeddings generiert wurden
        num_before = len(self.merged_input)
        self.merged_input = self.merged_input.dropna(subset=[col for col in embedding_df.columns if col.startswith("kmer_")])
        num_after = len(self.merged_input)

        logger.info(f"❌ Entferne {num_before - num_after} Samples ohne Kmer-Embeddings (NaN in Kmer-Spalten)")

        self.merged_input.fillna(0, inplace=True)

        new_kmer_cols = [col for col in embedding_df.columns if col.startswith("kmer_")]
        feature_cols_merged += new_kmer_cols

        #logger.info(f"✅ Final merged_input shape after k-mers: {self.merged_input.shape}")
        logger.info(self.merged_input.head(4))
        logger.info(f"🧪 Total features (excluding targets): {len(feature_cols_merged)}")
        logger.info(f"🎯 Total targets: {num_phenotype_cols - 2}")
        logger.info(f"🧾 Feature column sample: {feature_cols_merged} ")

        # 🔁 Sicheres Mapping erneut durchführen, nur falls nötig
        logger.info("🔁 Überprüfe und mappe Zielspalten nach der Kmer-Generierung...")
        for col in self.merged_input.columns[2:num_phenotype_cols]:
            if self.merged_input[col].dtype == object or self.merged_input[col].dtype.name == "category":
                logger.info(f"➡️  Mapping von Spalte '{col}'")
                self.merged_input[col] = (
                    self.merged_input[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace("NAN", "NA")
                    .map(RESISTANCE_MAPPING)
                    .fillna(0)
                    .astype(int)
                )


        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            self.merged_input.columns[2:num_phenotype_cols],
            feature_cols_merged,
            organism_mapping=organism_mapping,
        )

        logger.info(f"🎯 Target column names:\n{list(preprocessed_data.target_cols)}")

        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            logger.info("🧪 Preview of target values:\n%s", preprocessed_data.merged_input[preprocessed_data.target_cols].head(10))



        logger.info(f"✅ Final number of merged samples:{len(preprocessed_data.merged_input)}")
        logger.info(f"✅ Final merged_input shape:{preprocessed_data.merged_input.shape}")



        print(
            "Number of preprocessed merged samples: "
            + str(len(preprocessed_data.merged_input))
        )

        return preprocessed_data


@dataclass
class PreprocessedDataDTO:

    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
    organism_mapping: dict = field(default_factory=dict)
