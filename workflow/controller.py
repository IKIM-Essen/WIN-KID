import argparse
from collections import Counter, defaultdict
from datetime import datetime
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from copy import deepcopy
import os
import config
import preprocessing
import classic_rf
import stacked_rf
import constants
import utils
import shutil
from execution_modes import ExecutionMode, W2VMode
from constants import (
    CLASSIC_RF_SETTINGS,
    MODEL_FOLDER,
    ORGANISM_COLUMN,
    STACKED_RF_SETTINGS,
    RESISTANCE_MAPPING,
    W2V_SETTINGS,
)

from log import setup_logging

ANTIBIOTIC_COLUMN = "Antibiotic"
logger = logging.getLogger(__name__)
tuning_logger = logging.getLogger("w2v_tuning")


def log_run_configuration():
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUN CONFIGURATION")
    logger.info("=" * 70)

    logger.info(f"Execution mode : {config.EXECUTION_MODE.value}")
    logger.info(f"Feature mode   : {config.FEATURE_MODE.value}")
    logger.info(f"Stack model    : {config.STACK_MODEL}")
    logger.info(f"Cross validate : {config.CROSS_VALIDATE}")
    logger.info(f"PCA : {config.USE_PCA}")
    logger.info(f"PCA components : {config.PCA_COMPONENTS}")

    logger.info("")
    logger.info("Word2Vec settings")
    logger.info(f"  k-mer size        : {W2V_SETTINGS.k_size}")
    logger.info(f"  vector size       : {W2V_SETTINGS.vec_size}")
    logger.info(f"  window            : {W2V_SETTINGS.window}")
    logger.info(f"  skip-gram (sg)    : {W2V_SETTINGS.sg}")
    logger.info(f"  negative samples  : {W2V_SETTINGS.negative}")
    logger.info(f"  min_count         : {W2V_SETTINGS.min_count}")
    logger.info(f"  epochs            : {W2V_SETTINGS.w2v_epochs}")
    logger.info(f"  include_position  : {W2V_SETTINGS.include_position}")

    logger.info("")
    logger.info("Random Forest settings")
    logger.info(f"  n_estimators      : {CLASSIC_RF_SETTINGS.n_estimators}")
    logger.info(f"  max_depth         : {CLASSIC_RF_SETTINGS.max_depth}")
    logger.info(f"  max_features      : {CLASSIC_RF_SETTINGS.max_features}")

    logger.info("=" * 70)
    logger.info("")


def tune_w2v_via_stacked_rf(dataset_list_input, n_iter=5, random_state=42):
    """
    Tune Word2Vec hyperparameters based on final stacked Random Forest CV performance.
    """

    logger.info(
        "🔍 Starting task-driven Word2Vec hyperparameter tuning (Stacked RF based)"
    )

    # --- Define search space ---
    search_space = {
        "vec_size": [50, 100, 200],
        "window": [5, 15, 25],
        "negative": [5, 15, 25],
        "sg": [0, 1],
        "min_count": [2, 5, 8],
    }

    configs = list(
        ParameterSampler(search_space, n_iter=n_iter, random_state=random_state)
    )

    tuning_logger.info(f"🔎 Evaluating {len(configs)} Word2Vec configurations")

    best_score = -np.inf
    best_cfg = None

    # Store original settings to restore later
    original_settings = deepcopy(W2V_SETTINGS)

    for idx, cfg in enumerate(configs):
        tuning_logger.info(f"\n⚙️  Configuration {idx+1}/{len(configs)}: {cfg}")

        # --- Apply config temporarily ---
        tuned_settings = deepcopy(original_settings)
        for k, v in cfg.items():
            setattr(tuned_settings, k, v)

        try:
            # --- Re-run preprocessing with new W2V config ---
            data_loader = preprocessing.DataLoader()
            preprocessed_data = data_loader.get_preprocessed_data(
                dataset_list_input, tuned_settings
            )

            # --- Run stacked RF cross-validation ---
            (
                rf_results,
                _,
                _,
                _,
                _,
            ) = run_stacked_rf_cv(preprocessed_data)

            # --- Compute global macro-F1 score ---
            fold_scores = []

            for fold in rf_results:
                for target, result in fold.items():

                    if hasattr(result, "f1") and isinstance(result.f1, dict):
                        class_f1_scores = list(result.f1.values())

                        # Avoid empty dict
                        if len(class_f1_scores) > 0:
                            macro_f1 = float(np.mean(class_f1_scores))
                            fold_scores.append(macro_f1)
                        else:
                            tuning_logger.warning(f"Empty F1 dict for target {target}")
                    else:
                        tuning_logger.warning(f"No valid F1 found for target {target}")

            if not fold_scores:
                tuning_logger.warning(
                    "No valid F1 scores collected — skipping configuration"
                )
                continue

            mean_score = float(np.mean(fold_scores))

            tuning_logger.info(f"   → Mean CV Macro-F1: {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_cfg = cfg
                tuning_logger.info("   🏆 New best configuration found!")

        except Exception as e:
            tuning_logger.exception(f"❌ Configuration failed due to error: {e}")
            continue

    if best_cfg is None:
        raise RuntimeError("No valid Word2Vec configuration found during tuning.")

    tuning_logger.info("\n======================================")
    tuning_logger.info(f"🏆 Best W2V configuration: {best_cfg}")
    tuning_logger.info(f"🏆 Best CV Macro-F1: {best_score:.4f}")
    tuning_logger.info("======================================\n")
    logger.info(
        "Finished task-driven Word2Vec hyperparameter tuning (Stacked RF based)"
    )

    return best_cfg


def process(dataset_list_input):

    log_run_configuration()

    if config.W2V_MODE == W2VMode.TUNE_W2V:
        best_cfg = tune_w2v_via_stacked_rf(dataset_list_input, n_iter=5)

        # Apply best configuration permanently
        for k, v in best_cfg.items():
            setattr(W2V_SETTINGS, k, v)

        logger.info("🔁 Re-running full pipeline with best W2V configuration")

    # Preprocessing
    data_loader = preprocessing.DataLoader()
    if config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
        preprocessed_data_input = data_loader.get_genotype_data_for_prediction(
            dataset_list_input, W2V_SETTINGS
        )
    elif config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE:
        preprocessed_data_input = data_loader.get_genotype_data_for_prediction(
            dataset_list_input, W2V_SETTINGS
        )
    else:
        preprocessed_data_input = data_loader.get_preprocessed_data(
            dataset_list_input, W2V_SETTINGS
        )
        preprocessed_data_input = preprocessing.filter_merged_input(
            preprocessed_data_input, config.MIN_SAMPLE_NUMBER
        )

    start_time = time.time()
    # Compute

    if config.STACK_MODEL and not config.CROSS_VALIDATE:
        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
            per_organism_count,
        ) = run_stacked_rf(dataset_list_input, preprocessed_data_input)

    elif config.STACK_MODEL and config.CROSS_VALIDATE:
        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
            per_organism_count,
        ) = run_stacked_rf_cv(preprocessed_data_input)

    else:
        raise ValueError("Model strategy is invalid")

    # Save Evaluation
    if config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_SAVE:
        utils.per_organism_evaluation_to_csv(per_organism_results)
        utils.evaluation_to_csv(rf_results, y_test_count_results, y_train_count_results)
        if config.STACK_MODEL:
            utils.feature_importance_to_csv(rf_results)
            per_organism_count.to_csv("Evaluation/Per_Organism_Evaluation_count.csv")
    logger.info("--- %s seconds for ML---", (time.time() - start_time))


def run_stacked_rf(dataset_list_input, preprocessed_data_input):
    if config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED and os.path.exists(
        MODEL_FOLDER
    ):
        shutil.rmtree(MODEL_FOLDER)
        print("Folder removed:", MODEL_FOLDER)
    # Prepare for first layer
    merged_filtered_input, X, y = stacked_rf.prepare_first_layer(
        preprocessed_data_input
    )
    X_train, X_test, y_train, y_test = stacked_rf.split_stacked_rf(X, y)

    count_per_organism = merge_count_per_org(
        preprocessed_data_input, X_train, X_test, y_train, y_test
    )

    # Run first and second layer
    # fmt: off
    (
        y_train_count_results,
        y_test_count_results,
        rf_results,
        per_organism_results
    ) = compute_stacked_rf(
        preprocessed_data_input,
        merged_filtered_input,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    # fmt: on

    # Add overall results and evaluate
    rf_results = [rf_results]
    y_test_count_results = [y_test_count_results]
    y_train_count_results = [y_train_count_results]
    if config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open((MODEL_FOLDER + "run_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Run executed at: {now}\n")

    if config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
        utils.save_prediction_results(
            dataset_list_input, preprocessed_data_input, rf_results
        )

    if config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_SAVE:
        utils.display_results(rf_results[0])
        utils.feature_importance_to_csv(rf_results)

    return (
        rf_results,
        per_organism_results,
        y_test_count_results,
        y_train_count_results,
        count_per_organism,
    )


def run_stacked_rf_cv(preprocessed_data_input):
    rf_results, y_test_count_results, y_train_count_results, count_per_org_list = (
        [],
        [],
        [],
        [],
    )
    per_organism_results = {}
    fold_per_organism_results = defaultdict(list)
    # Prepare for first layer
    merged_filtered_input, X, y = stacked_rf.prepare_first_layer(
        preprocessed_data_input
    )
    if config.EXECUTION_MODE != ExecutionMode.TRAIN_TEST:
        raise ValueError("--mode shall be TRAIN_TEST for cross validation")

    # Split
    skf, stratify_col = stacked_rf.get_stratified_split(preprocessed_data_input)

    # Run first and second layer
    for train_idx, test_idx in skf.split(X, stratify_col):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        count_per_org_list.append(
            merge_count_per_org(
                preprocessed_data_input, X_train, X_test, y_train, y_test
            )
        )

        (
            y_train_count,
            y_test_count,
            result_dic,
            fold_per_organism_result,
        ) = compute_stacked_rf(
            preprocessed_data_input,
            merged_filtered_input,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # Add overall results and evaluate
        rf_results.append(result_dic)
        y_test_count_results.append(y_test_count)
        y_train_count_results.append(y_train_count)

        # Add per organism result
        for target, per_target_result in fold_per_organism_result.items():
            fold_per_organism_results[target].append(per_target_result)
    for target, per_target_results in fold_per_organism_results.items():
        per_organism_results[target] = utils.average_per_organism_results(
            per_target_results
        )

    count_per_org_combined = pd.concat(count_per_org_list, ignore_index=True)
    count_cols = [
        c
        for c in count_per_org_combined.columns
        if c not in [ORGANISM_COLUMN, ANTIBIOTIC_COLUMN]
    ]
    per_org_count_averaged = count_per_org_combined.groupby(
        [ORGANISM_COLUMN, ANTIBIOTIC_COLUMN], as_index=False
    )[count_cols].mean()

    return (
        rf_results,
        per_organism_results,
        y_test_count_results,
        y_train_count_results,
        per_org_count_averaged,
    )


def merge_count_per_org(preprocessed_data_input, X_train, X_test, y_train, y_test):
    counts_test = count_per_org(
        X_test, y_test, preprocessed_data_input.organism_mapping
    )
    counts_train = count_per_org(
        X_train, y_train, preprocessed_data_input.organism_mapping
    )

    train_renamed = counts_train.rename(
        columns={c: f"Train_Count_{c}" for c in RESISTANCE_MAPPING}
    )
    test_renamed = counts_test.rename(
        columns={c: f"Test_Count_{c}" for c in RESISTANCE_MAPPING}
    )

    merged_amr = test_renamed.merge(
        train_renamed,
        on=[ORGANISM_COLUMN, ANTIBIOTIC_COLUMN],
        how="inner",
    )

    return merged_amr


def count_per_org(X_input, y_input, organism_mapping):
    count = y_input.copy()
    count[ORGANISM_COLUMN] = X_input[ORGANISM_COLUMN]
    long_df = count.melt(
        id_vars=ORGANISM_COLUMN,
        value_vars=y_input.columns,
        var_name=ANTIBIOTIC_COLUMN,
        value_name="AMR_Result",
    )
    inv_map = {v: k for k, v in RESISTANCE_MAPPING.items()}
    long_df["AMR_Result"] = long_df["AMR_Result"].map(inv_map)
    counts = (
        long_df.dropna(subset=["AMR_Result"])
        .groupby([ORGANISM_COLUMN, ANTIBIOTIC_COLUMN, "AMR_Result"])
        .size()
        .reset_index(name="count")
    )
    amr_table = counts.pivot_table(
        index=[ORGANISM_COLUMN, ANTIBIOTIC_COLUMN],
        columns="AMR_Result",
        values="count",
        fill_value=0,
    ).reset_index()

    expected_cols = list(RESISTANCE_MAPPING.keys())
    amr_table = amr_table.reindex(
        columns=[ORGANISM_COLUMN, ANTIBIOTIC_COLUMN] + expected_cols,
        fill_value=0,
    )

    amr_table[ORGANISM_COLUMN] = amr_table[ORGANISM_COLUMN].map(organism_mapping)

    return amr_table


def compute_stacked_rf(
    preprocessed_data_input, merged_filtered_input, X_train, X_test, y_train, y_test
):
    (
        y_train_count_results,
        y_test_count_results,
        rf_results,
        per_organism_results,
    ) = ({}, {}, {}, {})

    proba_train_joined = pd.DataFrame([])
    proba_test_joined = pd.DataFrame([])

    # first layer
    proba_train_joined, proba_test_joined = stacked_rf.run_first_layer(
        preprocessed_data_input,
        CLASSIC_RF_SETTINGS,
        y_test,
        y_train,
        X_test,
        X_train,
        proba_train_joined,
        y_train_count_results,
        y_test_count_results,
    )

    # Second layer
    (
        y_proba_train,
        X_proba_train,
        y_proba_test,
        X_proba_test,
        organism_test,
    ) = stacked_rf.prepare_second_layer_data(
        preprocessed_data_input,
        merged_filtered_input,
        proba_train_joined,
        proba_test_joined,
    )
    for target in preprocessed_data_input.target_cols:
        # Filter out NA values
        (
            X_train_filtered,
            y_train_filtered,
            X_test_filtered,
            y_test_filtered,
            organism_test_filtered,
        ) = stacked_rf.filter_na_values(
            y_proba_train,
            X_proba_train,
            y_proba_test,
            X_proba_test,
            organism_test,
            target,
        )

        second_layer_result = classic_rf.run_random_forest(
            X_train_filtered,
            y_train_filtered,
            X_test_filtered,
            y_test_filtered,
            target,
            STACKED_RF_SETTINGS,
        )

        # Add results and evaluate
        rf_results[target] = second_layer_result
        y_pred = second_layer_result.y_pred
        per_organism_perf = utils.evaluate_per_organism(
            y_pred,
            y_test_filtered,
            organism_test_filtered,
            second_layer_result.y_score,
            target,
        )

        # Mapping organism name to string
        organism_mapping = preprocessed_data_input.organism_mapping
        per_organism_results[target] = {
            organism_mapping.get(org_code, f"Unknown ({org_code})"): metrics
            for org_code, metrics in per_organism_perf.items()
        }

    return y_train_count_results, y_test_count_results, rf_results, per_organism_results


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logfile = f"Evaluation/logs/run_{timestamp}.log"

    # Create log directory if it does not exist
    log_dir = os.path.dirname(logfile)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    setup_logging(
        logfile=logfile,
        console_level=config.LOGGING_LEVEL,
        file_level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Run stackPred on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")

    args = parser.parse_args()

    dataset_list = utils.load_dataset_paths(args.path_file)

    process(dataset_list)
