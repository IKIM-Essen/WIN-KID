# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from dataclasses import dataclass
import argparse
import os
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
)
import preprocessing

from constants import RESISTANCE_MAPPING


@dataclass
class ResultDTO:
    fpr: dict
    tpr: dict
    roc_auc: dict
    pr_auc: dict
    accuracy: float
    y_pred: np.ndarray
    y_score: np.ndarray
    feature_importance: dict


def generate_results(target_cols, y_test_results, y_score, y_pred_list, feat_import):
    result_dic = {}
    # TODO: fix:
    # /projects/envs/conda/jzander/envs/WIN-KID_env/lib/python3.13/site-packages/sklearn/metrics/_ranking.py:1188: UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
    # warnings.warn(
    # /projects/envs/conda/jzander/envs/WIN-KID_env/lib/python3.13/site-packages/sklearn/metrics/_ranking.py:1033: UserWarning: No positive class found in y_true, recall is set to one for all thresholds.
    for col_index, col in enumerate(target_cols):
        y_test_col = y_test_results[col_index]
        y_pred_col = y_pred_list[col_index]
        y_score_col = y_score[col_index]

        unique_classes = np.unique(y_test_col)

        fpr = {}
        tpr = {}
        roc_auc = {}
        pr_auc = {}

        unique_classes = [1, 2, 3]
        for label_class in unique_classes:
            y_test_binarized = (y_test_col == label_class).astype(int)
            if label_class >= y_score_col.shape[1]:  # Safety check
                print(f"Skipping class {label_class} for {col}, not in predictions")
                continue
            y_score_col = np.array(y_score_col)
            fpr[label_class], tpr[label_class], _ = roc_curve(
                y_test_binarized, y_score_col[:, label_class]
            )
            roc_auc[label_class] = auc(fpr[label_class], tpr[label_class])
            pr_auc[label_class] = average_precision_score(
                y_test_binarized, y_score_col[:, label_class]
            )

        accuracy = accuracy_score(y_test_col, y_pred_col)

        result_dic[col] = ResultDTO(
            fpr,
            tpr,
            roc_auc,
            pr_auc,
            accuracy,
            y_pred_col,
            y_score_col,
            feat_import[col_index],
        )

    return result_dic


def run_random_forest(preprocessed_data):

    rf_models = {}
    y_pred_list = []
    y_score_list = []
    feature_importance_list = []
    y_test_list = []
    target_cols = preprocessed_data.target_cols
    for col in preprocessed_data.target_cols:
        # TODO: Exclude classes with low count (Often I)
        merged_filtered_input = preprocessed_data.merged_input
        merged_filtered_input = merged_filtered_input[merged_filtered_input[col] != 0]
        X = merged_filtered_input[preprocessed_data.feature_cols]
        y = merged_filtered_input[col]

        # TODO Cluster train/test Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )
        y_test_list.append(y_test)

        model = RandomForestClassifier(
            n_estimators=10, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        y_pred_list.append(model.predict(X_test))

        # Ensure consistent ordering of probabilities (0=S, 1=I, 2=R)
        unique_classes = model.classes_  # Extract classes learned by RF
        y_proba = model.predict_proba(X_test)

        # Create a full 4-column probability array, filling missing classes with 0
        proba_full = np.zeros((y_proba.shape[0], 4))  # Shape (samples, 3 classes)
        for idx, class_label in enumerate(unique_classes):
            proba_full[:, class_label] = y_proba[:, idx]  # Map existing probabilities

        y_score_list.append(proba_full)  # Store correctly ordered probabilities

        rf_models[col] = model

        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=X_train.columns)
        feature_importance_list.append(forest_importances.sort_values(ascending=False))

        print(f"Label distribution for {col}:")
        print(y_test.value_counts())

    return (
        generate_results(
            target_cols, y_test_list, y_score_list, y_pred_list, feature_importance_list
        ),
        y_test_list,
    )


def load_dataset_paths(path_file):
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Settings file '{path_file}' not found.")

    path_df = pd.read_csv(path_file)

    if not {"DataSetName", "PathToCsv", "PathToGff"}.issubset(path_df.columns):
        raise ValueError(
            "Settings file must contain columns: DataSetName, PathToCsv, PathToGff"
        )

    return path_df


def display_results(results_dto, print_feat_imp):
    reverse_mapping = {v: k for k, v in RESISTANCE_MAPPING.items()}
    with PdfPages("rf_roc_report.pdf") as pdf:
        for name in results_dto:
            result = results_dto[name]
            print(f"{name}    Accuracy: {result.accuracy}")

            # Investigate feature importance
            if print_feat_imp:
                print("Ranked Feature Importance:")
                print(result.feature_importance)
                value_sum = 0.0
                counter = 0
                target_value_sum = 0.95  # Max is 1.0
                for value in result.feature_importance:
                    value_sum = value_sum + value
                    counter = counter + 1
                    if value_sum > target_value_sum:
                        print(
                            "Top "
                            + str(counter)
                            + " of "
                            + str(len(result.feature_importance))
                            + " Features needed for an Impact of "
                            + str(target_value_sum)
                        )
                        break
            for class_label in result.roc_auc:
                if np.isnan(result.roc_auc[class_label]):
                    continue

                class_name = reverse_mapping.get(class_label, str(class_label))

                print(f"  Class {class_name} ROC AUC: {result.roc_auc[class_label]}")

                display = RocCurveDisplay(
                    fpr=result.fpr[class_label],
                    tpr=result.tpr[class_label],
                    roc_auc=result.roc_auc[class_label],
                    estimator_name=f"RF-{class_name}",
                )
                fig = display.plot().figure_
                fig.suptitle(f"ROC - {name} (Class {class_name})")

                pdf.savefig(fig)
                plt.close(fig)


def evaluation_to_csv(results_dto, y_test_input):
    evaluation_df = pd.DataFrame(
        columns=[
            "Accuracy",
            "ROC_Mean",
            "PR_Mean",
            "Test_Count_S",
            "Test_Count_I",
            "Test_Count_R",
        ]
    )
    counter = 0
    for name in results_dto:
        result = results_dto[name]
        label_counts = y_test_input[counter].value_counts()
        count_s = label_counts.get(1, 0)
        count_i = label_counts.get(2, 0)
        count_r = label_counts.get(3, 0)
        evaluation_df.loc[name] = [
            result.accuracy,
            np.nanmean(list(result.roc_auc.values())),
            np.nanmean(list(result.pr_auc.values())),
            count_s,
            count_i,
            count_r,
        ]
        counter = counter + 1
    evaluation_df.loc["Median"] = (
        [statistics.median(evaluation_df["Accuracy"].values)]
        + [statistics.median(evaluation_df["ROC_Mean"].values)]
        + [statistics.median(evaluation_df["PR_Mean"].values)]
        + [pd.NA]
        + [pd.NA]
        + [pd.NA]
    )
    print(evaluation_df)
    evaluation_df[["Test_Count_S", "Test_Count_I", "Test_Count_R"]] = evaluation_df[
        ["Test_Count_S", "Test_Count_I", "Test_Count_R"]
    ].astype("Int64")
    os.makedirs("Evaluation", exist_ok=True)
    evaluation_df.to_csv("Evaluation/Evaluation.csv", index=True, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RF on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")
    args = parser.parse_args()

    dataset_list = load_dataset_paths(args.path_file)

    data_loader = preprocessing.DataLoader()
    preprocessed_data_input = data_loader.get_preprocessed_data(dataset_list)

    rf_results, y_test_output = run_random_forest(preprocessed_data_input)

    display_results(rf_results, False)
    evaluation_to_csv(rf_results, y_test_output)
