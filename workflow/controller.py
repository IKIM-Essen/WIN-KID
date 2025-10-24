import argparse
import os
import time

import pandas as pd

import config, preprocessing, random_forest
from execution_modes import ExecutionMode


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
    rf_settings_input = random_forest.get_default_rf_settings()

    rf_settings_stacked = random_forest.get_stacked_rf_settings()

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
            preprocessed_data_input, 15
        )

    start_time = time.time()
    if config.EXECUTION_MODE == ExecutionMode.TUNE_HYPERPARAMETER:
        random_forest.tune_hyperparameter(preprocessed_data_input)

    else:
        if not config.STACK_MODEL and not config.CROSS_VALIDATE:
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
                per_organism_results,
            ) = random_forest.run_splitted_random_forest(
                preprocessed_data_input,
                rf_settings_input,
            )

            random_forest.display_results(rf_results[0], False)

        elif not config.STACK_MODEL and config.CROSS_VALIDATE:
            # display not possible with CV.
            # S/R/I Set changes with every fold -> fpr size changes as well
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
                per_organism_results,
            ) = random_forest.run_cross_validated_random_forest(
                preprocessed_data_input,
                rf_settings_input,
            )

        elif config.STACK_MODEL and not config.CROSS_VALIDATE:
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
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
                random_forest.display_results(rf_results[0], False)
                random_forest.feature_importance_to_csv(rf_results)

        elif config.STACK_MODEL and config.CROSS_VALIDATE:
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
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
                rf_results, y_test_count_result, y_train_count_result
            )
            if config.STACK_MODEL:
                random_forest.feature_importance_to_csv(rf_results)
        print("--- %s seconds for ML---" % (time.time() - start_time))


# TODO: Add logger
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run stackPred on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")

    args = parser.parse_args()

    dataset_list = load_dataset_paths(args.path_file)

    process(dataset_list)
