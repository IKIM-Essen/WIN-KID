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


def generate_results(y_test, y_score, y_pred, feat_import):
    result_dic = {}

    for col_index, col in enumerate(y_test.columns):
        y_test_col = y_test[col]
        y_pred_col = y_pred[:, col_index]
        y_score_col = y_score[col_index]

        unique_classes = np.unique(y_test_col)

        fpr = {}
        tpr = {}
        roc_auc = {}
        pr_auc = {}

        for label_class in unique_classes:
            y_test_binarized = (y_test_col == label_class).astype(int)
            if label_class >= y_score_col.shape[1]:  # Safety check
                print(f"Skipping class {label_class} for {col}, not in predictions")
                continue

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

    X = preprocessed_data.merged_input[preprocessed_data.feature_cols]
    y = preprocessed_data.merged_input[preprocessed_data.target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    rf_models = {}
    y_pred_list = []
    y_score_list = []
    feature_importance_list = []

    for col in y_train.columns:
        model = RandomForestClassifier(
            n_estimators=10, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train[col])

        y_pred_list.append(model.predict(X_test))

        # Ensure consistent ordering of probabilities (0=S, 1=I, 2=R)
        unique_classes = model.classes_  # Extract classes learned by RF
        y_proba = model.predict_proba(X_test)

        # Create a full 3-column probability array, filling missing classes with 0
        proba_full = np.zeros((y_proba.shape[0], 3))  # Shape (samples, 3 classes)
        for idx, class_label in enumerate(unique_classes):
            proba_full[:, class_label] = y_proba[:, idx]  # Map existing probabilities

        y_score_list.append(proba_full)  # Store correctly ordered probabilities

        rf_models[col] = model

        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=X_train.columns)
        feature_importance_list.append(forest_importances.sort_values(ascending=False))

    for col in y_test.columns:
        print(f"Label distribution for {col}:")
        print(y_test[col].value_counts())

    y_pred = np.array(y_pred_list).T
    return (
        generate_results(y_test, y_score_list, y_pred, feature_importance_list),
        y_test,
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


def display_results(results_dto, print_feat_imp, y_test_input):
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

            label_counts = y_test_input[name].value_counts()
            count_s = label_counts.get(0, 0)
            count_i = label_counts.get(1, 0)
            count_r = label_counts.get(2, 0)
            evaluation_df.loc[name] = [
                result.accuracy,
                np.array(list(result.roc_auc.values())).mean(),
                np.array(list(result.pr_auc.values())).mean(),
                count_s,
                count_i,
                count_r,
            ]
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
        print(statistics.median(evaluation_df["Accuracy"].values))
        evaluation_df.loc["Median"] = (
            [statistics.median(evaluation_df["Accuracy"].values)]
            + [statistics.median(evaluation_df["ROC_Mean"].values)]
            + [statistics.median(evaluation_df["PR_Mean"].values)]
            + [pd.NA]
            + [pd.NA]
            + [pd.NA]
        )
        evaluation_df[["Test_Count_S", "Test_Count_I", "Test_Count_R"]] = evaluation_df[
            ["Test_Count_S", "Test_Count_I", "Test_Count_R"]
        ].astype("Int64")
        print(evaluation_df)
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

    rf_results, y_test = run_random_forest(preprocessed_data_input)

    # TODO Split display and save?
    display_results(rf_results, False, y_test)
