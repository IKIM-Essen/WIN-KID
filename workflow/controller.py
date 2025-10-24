import argparse
from collections import Counter, defaultdict
import os
import time
import logging

import pandas as pd

import config, preprocessing, random_forest
from execution_modes import ExecutionMode
from constants import ORGANISM_COLUMN


# TODO: Outsource to util?
def load_dataset_paths(path_file):
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Settings file '{path_file}' not found.")

    path_df = pd.read_csv(path_file)

    if not {"DataSetName", "PathToCsv", "PathToGff"}.issubset(path_df.columns):
        raise ValueError(
            "Settings file must contain columns: DataSetName, PathToCsv, PathToGff"
        )

    return path_df


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
        preprocessed_data_input = random_forest.filter_merged_input(
            preprocessed_data_input, config.MIN_SAMPLE_NUMBER
        )

    # Random Forest
    start_time = time.time()
    rf_settings_input = random_forest.get_default_rf_settings()
    rf_settings_stacked = random_forest.get_stacked_rf_settings()
    # if config.EXECUTION_MODE == ExecutionMode.TUNE_HYPERPARAMETER:
    #     random_forest.tune_hyperparameter(preprocessed_data_input)

    if not config.STACK_MODEL and not config.CROSS_VALIDATE:

        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
        ) = run_classic_rf(preprocessed_data_input)
        random_forest.display_results(rf_results[0])

    elif not config.STACK_MODEL and config.CROSS_VALIDATE:
        (
            rf_results,
            per_organism_results,
            y_test_count_results,
            y_train_count_results,
        ) = run_classic_rc_cv(preprocessed_data_input)

    elif config.STACK_MODEL and not config.CROSS_VALIDATE:
        (
            rf_results,
            y_test_count_results,
            y_train_count_results,
            per_organism_results,
        ) = random_forest.run_stacked_random_forest(
            preprocessed_data_input,
            rf_settings_input,
            rf_settings_stacked,
            False,
        )

        if config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
            random_forest.save_prediction_results(
                dataset_list_input,
                preprocessed_data_input,
                rf_results,
                dataset_list_input["PathToCsv"],
            )

        if config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_SAVE:
            random_forest.display_results(rf_results[0])
            random_forest.feature_importance_to_csv(rf_results)

    elif config.STACK_MODEL and config.CROSS_VALIDATE:
        (
            rf_results,
            y_test_count_results,
            y_train_count_results,
            per_organism_results,
        ) = random_forest.run_stacked_random_forest(
            preprocessed_data_input,
            rf_settings_input,
            rf_settings_stacked,
            True,
        )

    else:
        raise ValueError("Model strategy is invalid")
    if config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_SAVE:
        random_forest.per_organism_evaluation_to_csv(per_organism_results)
        random_forest.evaluation_to_csv(
            rf_results, y_test_count_results, y_train_count_results
        )
        if config.STACK_MODEL:
            random_forest.feature_importance_to_csv(rf_results)
    logging.info("--- %s seconds for ML---", (time.time() - start_time))


def run_classic_rf(preprocessed_data_input):
    y_test_count_result, y_train_count_result, rf_result = ({}, {}, {})
    per_organism_results = {}

    for col in preprocessed_data_input.target_cols:
        merged_filtered_input = preprocessed_data_input.merged_input
        merged_filtered_input = random_forest.filter_preprocessed_data(
            merged_filtered_input, col
        )
        y_test, y_train, X_test, X_train = random_forest.split_train_test(
            merged_filtered_input,
            merged_filtered_input[preprocessed_data_input.feature_cols],
            merged_filtered_input[col],
        )

        y_test_count_result[col] = Counter(y_test)
        y_train_count_result[col] = Counter(y_train)

        single_result = random_forest.run_random_forest(
            X_train,
            y_train,
            X_test,
            y_test,
            col,
            random_forest.get_default_rf_settings(),
        )
        rf_result[col] = single_result

        organism_test = merged_filtered_input.loc[y_test.index, ORGANISM_COLUMN]
        per_organism_results[col] = random_forest.evaluate_per_organism(
            single_result.y_pred, y_test, organism_test, single_result.y_score, col
        )

    return (
        [rf_result],
        per_organism_results,
        [y_test_count_result],
        [y_train_count_result],
    )


def run_classic_rc_cv(preprocessed_data_input):
    rf_results_return, y_test_count_results_return, y_train_count_results_return = (
        [],
        [],
        [],
    )
    per_organism_results_return = {}

    for col in preprocessed_data_input.target_cols:
        # Filter and prepare data
        merged_filtered_input = preprocessed_data_input.merged_input
        merged_filtered_input = random_forest.filter_preprocessed_data(
            merged_filtered_input, col
        )
        X = merged_filtered_input[preprocessed_data_input.feature_cols]
        y = merged_filtered_input[col]

        # Cross-validation splits
        split_iterator = random_forest.get_split_iterator(
            X,
            merged_filtered_input[ORGANISM_COLUMN],
        )
        fold_per_organism_results = defaultdict(list)

        # Iterate over folds
        for fold_idx, (train_idx, test_idx) in enumerate(split_iterator):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if set(y_train.unique()) != set(y_test.unique()):
                logging.warning(
                    "Skipped fold for %s: y_train and y_test have different classes.",
                    col,
                )
                continue

                # Train and predict
            result = random_forest.run_random_forest(
                X_train,
                y_train,
                X_test,
                y_test,
                col,
                random_forest.get_default_rf_settings(),
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
            per_fold_result = random_forest.evaluate_per_organism(
                result.y_pred, y_test, organism_test, result.y_score, col
            )
            fold_per_organism_results[col].append(per_fold_result)

        per_organism_results_return[col] = random_forest.average_per_organism_results(
            fold_per_organism_results[col]
        )

    return (
        rf_results_return,
        per_organism_results_return,
        y_test_count_results_return,
        y_train_count_results_return,
    )


# TODO: Add logger
if __name__ == "__main__":
    logging.basicConfig(level=config.LOGGING_LEVEL)
    parser = argparse.ArgumentParser(
        description="Run stackPred on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")

    args = parser.parse_args()

    dataset_list = load_dataset_paths(args.path_file)

    process(dataset_list)
