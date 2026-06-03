# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import re
import logging
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
from constants import W2V_MODEL_PATH
from constants import W2V_SETTINGS
from config import EXECUTION_MODE, KMERE, FASTA_DIR, RETRAIN_W2V, W2V_MODE
from execution_modes import ExecutionMode
import random
from kmers import train_word2vec_model_streaming, encode_all_samples, load_w2v_model

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

GFF_DIR = "resources/genotype"
MIN_SAMPLES_PER_ORG = 10

logger = logging.getLogger(__name__)


def extract_gene_attribute(attribute_string, key):
    pattern = rf"{key}=([^;]+)"
    match = re.search(pattern, attribute_string)
    return match.group(1) if match else None


def extract_single_features(gff_df_input, features):
    logging.info(gff_df_input.columns)

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

    logger.info(
        "%s input genotype samples from %s",
        str(len(data)),
        directory_row["DataSetName"],
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
            logger.warning(
                "⚠️ Similar columns: '%s' ↔ '%s' (Score: %s)",
                col1,
                {col2},
                score,
            )


def filter_merged_input(preprocessed_data, min_sample_number):

    merged_filtered_input = preprocessed_data.merged_input.copy()

    print("\n" + "=" * 80)
    print("[DEBUG] START FILTERING")
    print("=" * 80)

    print(f"[DEBUG] Initial dataframe shape: {merged_filtered_input.shape}")
    print(f"[DEBUG] Initial target count: {len(preprocessed_data.target_cols)}")

    print("\n[DEBUG] Initial target distributions (10 examples):")
    for col in preprocessed_data.target_cols[:10]:
        print(f"  {col}:")
        print(merged_filtered_input[col].value_counts(dropna=False))

    # ------------------------------------------------------------------
    # Iterate through targets
    # ------------------------------------------------------------------
    for idx, col_name in enumerate(preprocessed_data.target_cols):

        print("\n" + "-" * 80)
        print(f"[DEBUG] Processing target {idx+1}/{len(preprocessed_data.target_cols)}")
        print(f"[DEBUG] Column: {col_name}")
        print("-" * 80)

        before_shape = merged_filtered_input.shape[0]

        value_counts = merged_filtered_input[col_name].value_counts(dropna=False)

        print("[DEBUG] Value counts BEFORE filtering:")
        print(value_counts)

        low_freq_values = value_counts[value_counts < min_sample_number].index

        print(f"[DEBUG] min_sample_number = {min_sample_number}")
        print(f"[DEBUG] Low frequency values: {list(low_freq_values)}")

        if len(low_freq_values) > 0:

            rows_to_remove = merged_filtered_input[
                merged_filtered_input[col_name].isin(low_freq_values)
            ]

            print(f"[DEBUG] Rows to remove: {len(rows_to_remove)}")

            # Show sample rows
            print("[DEBUG] Example rows being removed:")
            print(rows_to_remove[[col_name]].head(10))
        # ------------------------------------------------------------------
        # FILTER
        # ------------------------------------------------------------------
        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input[col_name].isin(low_freq_values)
        ]

        after_shape = merged_filtered_input.shape[0]

        print(f"[DEBUG] Shape before: {before_shape}")
        print(f"[DEBUG] Shape after : {after_shape}")
        print(f"[DEBUG] Removed rows: {before_shape - after_shape}")

        updated_value_counts = merged_filtered_input[col_name].value_counts(
            dropna=False
        )

        # Remove NaN values from counter
        updated_value_counts = updated_value_counts.drop(labels=0, errors="ignore")

        print(f"[DEBUG] Remaining unique labels: {len(updated_value_counts)}")

        # ------------------------------------------------------------------
        # REMOVE TARGET IF ONLY ONE CLASS LEFT
        # ------------------------------------------------------------------
        if len(updated_value_counts) < 2:

            print(f"[DEBUG] DROPPING TARGET: {col_name}")

            merged_filtered_input = merged_filtered_input.drop(col_name, axis=1)

            logger.warning(
                "%s removed because only one class is left after filtering",
                col_name,
            )

            preprocessed_data.target_cols = preprocessed_data.target_cols.difference(
                [col_name]
            )

    print("\n" + "=" * 80)
    print("[DEBUG] REMOVING FULL-ZERO ROWS")
    print("=" * 80)

    print(f"[DEBUG] Shape before zero-row removal: {merged_filtered_input.shape}")

    mask = (merged_filtered_input[preprocessed_data.target_cols] != 0).any(axis=1)

    removed_zero_rows = (~mask).sum()

    print(f"[DEBUG] Rows with all-zero targets: {removed_zero_rows}")

    if removed_zero_rows > 0:
        print("[DEBUG] Example all-zero rows:")
        print(merged_filtered_input.loc[~mask].head(10))

    merged_filtered_input = merged_filtered_input[mask]

    print(f"[DEBUG] Final dataframe shape: {merged_filtered_input.shape}")

    # ----------------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[DEBUG] FINAL SUMMARY")
    print("=" * 80)

    print(f"[DEBUG] Remaining targets: {len(preprocessed_data.target_cols)}")

    print("[DEBUG] Remaining target names:")
    print(list(preprocessed_data.target_cols))

    if ORGANISM_COLUMN in merged_filtered_input.columns:
        print("\n[DEBUG] Organism distribution:")
        print(merged_filtered_input[ORGANISM_COLUMN].value_counts())

    preprocessed_data.merged_input = merged_filtered_input

    logger.info(
        "Number of samples after sample number filtering: %s",
        len(preprocessed_data.merged_input),
    )

    logger.info(preprocessed_data.merged_input[ORGANISM_COLUMN].value_counts())

    os.makedirs("Evaluation", exist_ok=True)

    preprocessed_data.merged_input[ID_COLUMN].to_csv(
        "Evaluation/samples_used.csv", index=False
    )

    return preprocessed_data


class DataLoader:
    def __init__(self):
        self.merged_input = None

    def _add_kmer_embeddings(self, feature_cols_merged, W2v_settings):
        if not KMERE:
            logger.info("KMERE disabled - skipping k-mer embedding step.")
            return feature_cols_merged

        fasta_ids = self.merged_input[ID_COLUMN].unique().tolist()

        missing = [
            sid
            for sid in fasta_ids
            if not os.path.exists(os.path.join(FASTA_DIR, f"{sid}.fna.gz"))
        ]

        logger.info(f"Missing FASTA files: {len(missing)} / {len(fasta_ids)}")
        logger.info(f"Total FASTA IDs available: {len(fasta_ids)}")
        logger.info(f"Sample FASTA IDs: {fasta_ids[:5]}")
        logger.info("W2V Mode: TRAIN_W2V")

        # Load or train model
        if (
            os.path.exists(W2V_MODEL_PATH)
            and not RETRAIN_W2V
            and not W2V_MODE == "TUNE_W2V"
        ):
            logger.info("📥 Loading existing Word2Vec model...")
            w2v_model = load_w2v_model(W2V_MODEL_PATH)

            eval_ids = fasta_ids
        else:
            logger.info("🧪 Training new Word2Vec model...")

            if W2V_MODE == "TUNE_W2V":
                random.seed(W2v_settings.seed)
                subset_size = max(1, int(len(fasta_ids) * 0.2))
                train_ids = random.sample(fasta_ids, min(subset_size, len(fasta_ids)))

                # use SAME subset for evaluation
                eval_ids = train_ids

                logger.info(f"⚡ Tuning mode: using {len(train_ids)} samples (20%)")

            else:
                train_ids = random.sample(
                    fasta_ids, min(W2v_settings.train_subset_size, len(fasta_ids))
                )

                eval_ids = fasta_ids

                logger.info(
                    f"Train Word2Vec on {len(train_ids)} FASTA files using streaming..."
                )

            w2v_model = train_word2vec_model_streaming(
                train_ids,
                FASTA_DIR,
                W2v_settings,
                save_path=W2V_MODEL_PATH,
            )

        if W2V_MODE == "TUNE_W2V":
            logger.info("⚡ Filtering dataset to subset for tuning")
            self.merged_input = self.merged_input[
                self.merged_input[ID_COLUMN].isin(eval_ids)
            ]

        logger.info("Generate aggregated k-mer embeddings...")
        embedding_df = encode_all_samples(eval_ids, FASTA_DIR, w2v_model, W2v_settings)

        if embedding_df is None:
            logger.warning("No k-mer embeddings generated.")

        logger.info("==== DEBUG: EMBEDDING_DF ====")
        logger.info(f"Shape: {embedding_df.shape}")
        logger.info(f"First row:\n{embedding_df.head(1)}")

        # Check ID overlap
        logger.info(f"Unique embedding IDs: {embedding_df[ID_COLUMN].nunique()}")
        logger.info(
            f"Unique merged_input IDs (before merge): {self.merged_input[ID_COLUMN].nunique()}"
        )

        intersection = set(embedding_df[ID_COLUMN]) & set(self.merged_input[ID_COLUMN])
        logger.info(f"ID intersection size: {len(intersection)}")

        if len(intersection) == 0:
            raise ValueError("No matching IDs between embeddings and merged_input!")

        logger.info(f"Embedding DataFrame shape: {embedding_df.shape}")

        kmer_cols = [col for col in embedding_df.columns if col.startswith("kmer_")]

        if not kmer_cols:
            raise ValueError("No k-mer embedding columns found in embedding_df")

        logger.info("==== DEBUG: KMER COLS (FROM EMBEDDING_DF) ====")
        logger.info(f"Detected kmer_cols count: {len(kmer_cols)}")
        logger.info(f"Sample kmer_cols: {kmer_cols[:10]}")

        # Merge embeddings
        self.merged_input = pd.merge(
            self.merged_input, embedding_df, on=ID_COLUMN, how="left"
        )

        logger.info("==== DEBUG: AFTER MERGE ====")
        logger.info(f"Merged shape: {self.merged_input.shape}")
        logger.info(f"Merged columns sample: {self.merged_input.columns.tolist()}")

        # Check if kmer columns exist
        existing_kmers = [col for col in kmer_cols if col in self.merged_input.columns]
        missing_kmers = [
            col for col in kmer_cols if col not in self.merged_input.columns
        ]

        logger.info(f"Existing kmer cols: {len(existing_kmers)}")
        logger.info(f"Missing kmer cols: {len(missing_kmers)}")

        if missing_kmers:
            logger.error(f"Missing kmer columns (first 10): {missing_kmers[:10]}")

        if existing_kmers:
            nan_counts = self.merged_input[existing_kmers].isna().sum().sum()
            logger.info(f"Total NaNs in kmer columns: {nan_counts}")

            logger.info("Sample kmer values:")
            logger.info(self.merged_input[existing_kmers].head(3))

        # Remove rows without embeddings
        num_before = len(self.merged_input)
        kmer_cols = [col for col in embedding_df.columns if col.startswith("kmer_")]

        logger.info("==== DEBUG: BEFORE DROPNA ====")
        logger.info(f"Trying to dropna on {len(kmer_cols)} columns")

        try:
            self.merged_input = self.merged_input.dropna(subset=kmer_cols)
        except Exception as e:
            logger.error("DROPNA FAILED")
            logger.error(f"Error: {e}")
            logger.error(f"kmer_cols (first 10): {kmer_cols[:10]}")
            raise
        num_after = len(self.merged_input)

        logger.info(
            f"❌ Removed {num_before - num_after} samples without k-mer embeddings"
        )

        self.merged_input.fillna(0, inplace=True)

        feature_cols_merged += kmer_cols

        return feature_cols_merged

    def preprocess_phenotype_data(self, dataset_list):
        seen_once = set()
        duplicates = set()
        input_phenotype_list = []
        for _, phenotype_file_row in dataset_list.iterrows():
            input_phenotype_data = pd.read_csv(phenotype_file_row["PathToCsv"])
            logger.info(
                "%s input phenotype samples from %s",
                str(len(input_phenotype_data)),
                phenotype_file_row["DataSetName"],
            )

            # Check for and remove duplicates inside dataset
            duplicate_ids = input_phenotype_data[ID_COLUMN][
                input_phenotype_data[ID_COLUMN].duplicated()
            ].unique()
            if len(duplicate_ids) > 0:
                logger.info(
                    "Removed %s duplicate IDs: %s",
                    len(duplicate_ids),
                    list(duplicate_ids),
                )
            else:
                logger.info("No duplicate IDs found.")
            input_phenotype_data = input_phenotype_data[
                ~input_phenotype_data[ID_COLUMN].isin(duplicate_ids)
            ]

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
        logger.info("%s input phenotype samples overall", str(len(input_phenotype)))

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

            if not updated or EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
                break

            if rows_to_remove:
                logger.info("🔻 Removing rows due to rare classes:")
                logger.info(input_phenotype.loc[sorted(rows_to_remove)])
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
        logger.info(
            "%s input genotype samples overall", str(len(input_genotype_combined))
        )

        return input_genotype_combined

    def get_genotype_data_for_prediction(self, dataset_list, W2v_settings):
        input_genotype = self.preprocess_genotype_data(dataset_list)

        # Load allowed features
        features = pd.read_csv("resources/settings/FeatureList.csv", header=None)
        all_features = features[0].tolist()
        all_features = [f for f in all_features if not f.startswith("kmer_")]

        # Always keep Sample_ID_IfH
        if "Sample_ID_IfH" not in all_features:
            all_features = ["Sample_ID_IfH"] + all_features

        # Reindex the dataframe: add missing columns (fill with 0), drop extra ones
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

        # Harmonise target antibiotics
        input_phenotype.columns = [
            (
                col
                if col == ID_COLUMN or col == ORGANISM_COLUMN or col.endswith("_AB")
                else f"{col}_AB"
            )
            for col in input_phenotype.columns
        ]
        for col in names:
            if col not in input_phenotype.columns:
                input_phenotype[col] = 0
        for col in input_phenotype.columns:
            if col not in names and col != ID_COLUMN:
                input_phenotype.drop(columns=[col])

        # Encode target values as specific ints
        for col in input_phenotype.columns[2:]:
            input_phenotype[col] = (
                input_phenotype[col].map(RESISTANCE_MAPPING).fillna(-1).astype(int)
            )

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

        feature_cols_merged = self._add_kmer_embeddings(
            feature_cols_merged, W2v_settings
        )

        # --- enforce feature alignment ---
        expected_features = pd.read_csv(
            "resources/settings/FeatureList.csv", header=None
        )[0].tolist()
        expected_features = [f for f in expected_features if not f.startswith("kmer_")]
        kmer_cols = [
            col for col in self.merged_input.columns if col.startswith("kmer_")
        ]
        final_features = expected_features + kmer_cols
        target_cols = [col for col in self.merged_input.columns if col.endswith("_AB")]
        final_columns = [ID_COLUMN] + target_cols + final_features

        self.merged_input = self.merged_input.reindex(
            columns=final_columns, fill_value=0
        )
        # update feature list used downstream
        feature_cols_merged = final_features

        logger.info(
            f"✅ Final merged_input shape after k-mers: {self.merged_input.shape}"
        )
        logger.info(self.merged_input.head(4))
        logger.info(f"Total features (excluding targets): {len(feature_cols_merged)}")
        logger.info(f"Total targets: {num_phenotype_cols - 2}")
        logger.info(f"Feature column sample: {feature_cols_merged} ")

        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            names,
            feature_cols_merged,
            organism_mapping=organism_mapping,
        )
        logger.info(
            "Total number of samples: %s", str(len(preprocessed_data.merged_input))
        )
        return preprocessed_data

    def get_preprocessed_data(self, dataset_list, W2v_settings):
        input_phenotype = self.preprocess_phenotype_data(dataset_list)
        input_genotype = self.preprocess_genotype_data(dataset_list)

        input_phenotype[ID_COLUMN] = input_phenotype[ID_COLUMN].astype(str).str.strip()
        input_genotype[ID_COLUMN] = input_genotype[ID_COLUMN].astype(str).str.strip()

        # Remove rare species, encode Organism Code as ints and save mapping
        counts = input_phenotype[ORGANISM_COLUMN].value_counts()
        input_phenotype = input_phenotype[
            input_phenotype[ORGANISM_COLUMN].isin(
                counts[counts >= MIN_SAMPLES_PER_ORG].index
            )
        ]
        removed_species = counts[counts < MIN_SAMPLES_PER_ORG]
        logger.info("Removed species (less than %s samples):", str(MIN_SAMPLES_PER_ORG))
        logger.info(removed_species)
        organism_cat = input_phenotype[ORGANISM_COLUMN].astype("category")
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

        feature_cols_merged = self._add_kmer_embeddings(
            feature_cols_merged, W2v_settings
        )

        # --- enforce feature alignment ---
        expected_features = pd.read_csv(
            "resources/settings/FeatureList.csv", header=None
        )[0].tolist()
        expected_features = [f for f in expected_features if not f.startswith("kmer_")]
        kmer_cols = [
            col for col in self.merged_input.columns if col.startswith("kmer_")
        ]
        final_features = expected_features + kmer_cols
        target_cols = [col for col in self.merged_input.columns if col.endswith("_AB")]
        final_columns = [ID_COLUMN] + target_cols + final_features

        self.merged_input = self.merged_input.reindex(
            columns=final_columns, fill_value=0
        )
        # update feature list used downstream
        feature_cols_merged = final_features

        logger.info(
            f"✅ Final merged_input shape after k-mers: {self.merged_input.shape}"
        )
        logger.info(self.merged_input.head(4))
        logger.info(f"Total features (excluding targets): {len(feature_cols_merged)}")
        logger.info(f"Total targets: {num_phenotype_cols - 2}")
        logger.info(f"Feature column sample: {feature_cols_merged} ")

        preprocessed_data = PreprocessedDataDTO(
            self.merged_input,
            [col for col in self.merged_input.columns if col.endswith("_AB")],
            feature_cols_merged,
            organism_mapping=organism_mapping,
        )

        if EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
            pd.DataFrame(preprocessed_data.feature_cols).to_csv(
                "resources/settings/FeatureList.csv", index=False, header=False
            )
        logger.info(
            "Total number of samples: %s", str(len(preprocessed_data.merged_input))
        )
        return preprocessed_data


@dataclass
class PreprocessedDataDTO:

    merged_input: pd.DataFrame
    target_cols: list
    feature_cols: list
    organism_mapping: dict = field(default_factory=dict)
