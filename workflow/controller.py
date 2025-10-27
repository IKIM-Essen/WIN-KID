import argparse
from collections import Counter, defaultdict
from datetime import datetime
import time
import logging

import pandas as pd

import config
import preprocessing
import classic_rf
import stacked_rf
import constants
import utils
from execution_modes import ExecutionMode
from constants import MODEL_FOLDER, ORGANISM_COLUMN

logger = logging.getLogger(__name__)


def process(dataset_list_input):

    # Preprocessing
    data_loader = preprocessing.DataLoader()
    if config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
        preprocessed_data_input = data_loader.get_genotype_data_for_prediction(
            dataset_list_input
        )
    elif config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE:
        preprocessed_data_input = data_loader.get_preprocessed_data(dataset_list_input)
    else:
        preprocessed_data_input = data_loader.get_preprocessed_data(dataset_list_input)
        preprocessed_data_input = preprocessing.filter_merged_input(
            preprocessed_data_input, config.MIN_SAMPLE_NUMBER
        )

    # Random Forest
    start_time = time.time()
    # if config.EXECUTION_MODE == ExecutionMode.TUNE_HYPERPARAMETER:
    #     random_forest.tune_hyperparameter(preprocessed_data_input)

    if not config.STACK_MODEL and not config.CROSS_VALIDATE:

        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
        ) = run_classic_rf(preprocessed_data_input)
        utils.display_results(rf_results[0])

    elif not config.STACK_MODEL and config.CROSS_VALIDATE:
        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
        ) = run_classic_rf_cv(preprocessed_data_input)

    elif config.STACK_MODEL and not config.CROSS_VALIDATE:
        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
        ) = run_stacked_rf(dataset_list_input, preprocessed_data_input)

    elif config.STACK_MODEL and config.CROSS_VALIDATE:
        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
        ) = run_stacked_rf_cv(preprocessed_data_input)

    else:
        raise ValueError("Model strategy is invalid")

    # Save Evaluation
    if config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_SAVE:
        utils.per_organism_evaluation_to_csv(per_organism_results)
        utils.evaluation_to_csv(rf_results, y_test_count_results, y_train_count_results)
        if config.STACK_MODEL:
            utils.feature_importance_to_csv(rf_results)
    logger.info("--- %s seconds for ML---", (time.time() - start_time))


def run_stacked_rf_cv(preprocessed_data_input):
    merged_filtered_input, X, y = stacked_rf.prepare_first_layer(
        preprocessed_data_input
    )
    if config.EXECUTION_MODE != ExecutionMode.TRAIN_TEST:
        raise ValueError("--mode shall be TRAIN_TEST for cross validation")

    skf, stratify_col = stacked_rf.get_stratified_split(preprocessed_data_input)

    rf_results = []
    y_test_count_results = []
    y_train_count_results = []
    per_organism_results = {}
    fold_per_organism_results = defaultdict(list)
    for train_idx, test_idx in skf.split(X, stratify_col):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

        rf_results.append(result_dic)
        y_test_count_results.append(y_test_count)
        y_train_count_results.append(y_train_count)

        for target, per_target_result in fold_per_organism_result.items():
            fold_per_organism_results[target].append(per_target_result)
    for target, per_target_results in fold_per_organism_results.items():
        per_organism_results[target] = utils.average_per_organism_results(
            per_target_results
        )

    return rf_results, per_organism_results, y_test_count_results, y_train_count_results


def run_stacked_rf(dataset_list_input, preprocessed_data_input):
    merged_filtered_input, X, y = stacked_rf.prepare_first_layer(
        preprocessed_data_input
    )
    X_train, X_test, y_train, y_test = stacked_rf.split_stacked_rf(X, y)

    y_train_count_results, y_test_count_results, rf_results, per_organism_results = (
        compute_stacked_rf(
            preprocessed_data_input,
            merged_filtered_input,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

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

    return rf_results, per_organism_results, y_test_count_results, y_train_count_results


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

    # FIRST Layer
    proba_train_joined, proba_test_joined = stacked_rf.run_first_layer(
        preprocessed_data_input,
        constants.CLASSIC_RF_SETTINGS,
        y_test,
        y_train,
        X_test,
        X_train,
        proba_train_joined,
        y_train_count_results,
        y_test_count_results,
    )

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

    # SECOND LAYER
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
            constants.STACKED_RF_SETTINGS,
        )
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


def run_classic_rf(preprocessed_data_input):
    y_test_count_result, y_train_count_result, rf_result = ({}, {}, {})
    per_organism_results = {}

    for col in preprocessed_data_input.target_cols:
        merged_filtered_input = preprocessed_data_input.merged_input
        merged_filtered_input = classic_rf.filter_preprocessed_data(
            merged_filtered_input, col
        )
        y_test, y_train, X_test, X_train = classic_rf.split_train_test(
            merged_filtered_input,
            merged_filtered_input[preprocessed_data_input.feature_cols],
            merged_filtered_input[col],
        )

        y_test_count_result[col] = Counter(y_test)
        y_train_count_result[col] = Counter(y_train)

        single_result = classic_rf.run_random_forest(
            X_train,
            y_train,
            X_test,
            y_test,
            col,
            constants.CLASSIC_RF_SETTINGS,
        )
        rf_result[col] = single_result

        organism_test = merged_filtered_input.loc[y_test.index, ORGANISM_COLUMN]
        per_organism_results[col] = utils.evaluate_per_organism(
            single_result.y_pred, y_test, organism_test, single_result.y_score, col
        )

    return (
        [rf_result],
        per_organism_results,
        [y_test_count_result],
        [y_train_count_result],
    )


def run_classic_rf_cv(preprocessed_data_input):
    rf_results_return, y_test_count_results_return, y_train_count_results_return = (
        [],
        [],
        [],
    )
    per_organism_results_return = {}

    for col in preprocessed_data_input.target_cols:
        # Filter and prepare data
        merged_filtered_input = preprocessed_data_input.merged_input
        merged_filtered_input = classic_rf.filter_preprocessed_data(
            merged_filtered_input, col
        )
        X = merged_filtered_input[preprocessed_data_input.feature_cols]
        y = merged_filtered_input[col]

        # Cross-validation splits
        split_iterator = classic_rf.get_split_iterator(
            X,
            merged_filtered_input[ORGANISM_COLUMN],
        )
        fold_per_organism_results = defaultdict(list)

        # Iterate over folds
        for fold_idx, (train_idx, test_idx) in enumerate(split_iterator):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if set(y_train.unique()) != set(y_test.unique()):
                logger.warning(
                    "Skipped fold for %s: y_train and y_test have different classes.",
                    col,
                )
                continue

                # Train and predict
            result = classic_rf.run_random_forest(
                X_train,
                y_train,
                X_test,
                y_test,
                col,
                constants.CLASSIC_RF_SETTINGS,
            )

            # Add overall results
            if len(rf_results_return) <= fold_idx:
                rf_results_return.append({})
                y_test_count_results_return.append({})
                y_train_count_results_return.append({})
            (rf_results_return[fold_idx])[col] = result
            (y_test_count_results_return[fold_idx])[col] = y_test.value_counts()
            (y_train_count_results_return[fold_idx])[col] = y_train.value_counts()

            # Add results per organism
            organism_test = merged_filtered_input.iloc[test_idx][ORGANISM_COLUMN]
            per_fold_result = utils.evaluate_per_organism(
                result.y_pred, y_test, organism_test, result.y_score, col
            )
            fold_per_organism_results[col].append(per_fold_result)

        per_organism_results_return[col] = utils.average_per_organism_results(
            fold_per_organism_results[col]
        )

    return (
        rf_results_return,
        per_organism_results_return,
        y_test_count_results_return,
        y_train_count_results_return,
    )


if __name__ == "__main__":
    logging.basicConfig(level=config.LOGGING_LEVEL)
    parser = argparse.ArgumentParser(
        description="Run stackPred on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")

    args = parser.parse_args()

    dataset_list = utils.load_dataset_paths(args.path_file)

    process(dataset_list)
