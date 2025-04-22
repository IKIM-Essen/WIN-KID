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
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.cluster import KMeans
from statistics import mean
from statistics import median
import preprocessing

from constants import RESISTANCE_MAPPING


@dataclass
class ResultDTO:
    fpr: dict
    tpr: dict
    roc_auc: dict
    pr_auc: dict
    accuracy: float
    precision: dict
    recall: dict
    f1: dict
    y_pred: np.ndarray
    y_score: np.ndarray
    feature_importance: dict


def generate_results(target_cols, y_test_results, y_score, y_pred_list, feat_import):
    result_dic = {}

    for col_index, col in enumerate(target_cols):
        y_test_col = y_test_results[col_index]
        y_pred_col = y_pred_list[col_index]
        y_score_col = y_score[col_index]

        unique_classes = np.unique(y_test_col)

        fpr = {}
        tpr = {}
        roc_auc = {}
        pr_auc = {}
        precision = {}
        recall = {}
        f1 = {}

        for label_class in unique_classes:
            y_test_binarized = (y_test_col == label_class).astype(int)
            y_pred_binarized = (y_pred_col == label_class).astype(int)

            if label_class >= y_score_col.shape[1]:
                print(f"Skipping class {label_class} for {col}, not in predictions")
                continue

            fpr[label_class], tpr[label_class], _ = roc_curve(
                y_test_binarized, y_score_col[:, label_class]
            )
            roc_auc[label_class] = auc(fpr[label_class], tpr[label_class])
            pr_auc[label_class] = average_precision_score(
                y_test_binarized, y_score_col[:, label_class]
            )

            precision[label_class] = precision_score(
                y_test_binarized, y_pred_binarized, zero_division=0
            )
            recall[label_class] = recall_score(
                y_test_binarized, y_pred_binarized, zero_division=0
            )
            f1[label_class] = f1_score(
                y_test_binarized, y_pred_binarized, zero_division=0
            )

        accuracy = accuracy_score(y_test_col, y_pred_col)

        result_dic[col] = ResultDTO(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            y_pred=y_pred_col,
            y_score=y_score_col,
            feature_importance=feat_import[col_index],
        )

    return result_dic


def generate_result(target_col, y_test_col, y_score_col, y_pred_col, feat_import):
    unique_classes = np.unique(y_test_col)

    fpr = {}
    tpr = {}
    roc_auc = {}
    pr_auc = {}
    precision = {}
    recall = {}
    f1 = {}

    for label_class in unique_classes:
        y_test_binarized = (y_test_col == label_class).astype(int)
        y_pred_binarized = (y_pred_col == label_class).astype(int)

        if label_class >= y_score_col.shape[1]:
            print(f"Skipping class {label_class} for {target_col}, not in predictions")
            continue

        fpr[label_class], tpr[label_class], _ = roc_curve(
            y_test_binarized, y_score_col[:, label_class]
        )
        roc_auc[label_class] = auc(fpr[label_class], tpr[label_class])
        pr_auc[label_class] = average_precision_score(
            y_test_binarized, y_score_col[:, label_class]
        )

        precision[label_class] = precision_score(
            y_test_binarized, y_pred_binarized, zero_division=0
        )
        recall[label_class] = recall_score(
            y_test_binarized, y_pred_binarized, zero_division=0
        )
        f1[label_class] = f1_score(y_test_binarized, y_pred_binarized, zero_division=0)

    accuracy = accuracy_score(y_test_col, y_pred_col)

    return ResultDTO(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        y_pred=y_pred_col,
        y_score=y_score_col,
        feature_importance=feat_import,
    )


def run_random_forest(preprocessed_data):

    y_pred_list = []
    y_score_list = []
    feature_importance_list = []
    y_test_list = []
    y_train_list = []
    target_cols = preprocessed_data.target_cols
    for col in preprocessed_data.target_cols:
        merged_filtered_input = preprocessed_data.merged_input
        merged_filtered_input = merged_filtered_input[merged_filtered_input[col] != 0]

        # Drop rows of Organisms that occur only once
        value_counts = merged_filtered_input["Organism_Code"].value_counts()
        rare_values = value_counts[value_counts == 1].index
        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input["Organism_Code"].isin(rare_values)
        ]

        X = merged_filtered_input[preprocessed_data.feature_cols]
        y = merged_filtered_input[col]

        # TODO: Why is that whorse
        # Clusterd split
        # cluster_labels = KMeans(
        #     n_clusters=int((len(merged_filtered_input) / 10)), random_state=42
        # ).fit_predict(X)
        # unique_clusters = np.unique(cluster_labels)
        # train_clusters, test_clusters = train_test_split(
        #     unique_clusters, test_size=0.5, random_state=42
        # )

        # train_idx = np.isin(cluster_labels, train_clusters)
        # test_idx = ~train_idx

        # X_train = X[train_idx]
        # X_test = X[test_idx]
        # y_train = y[train_idx]
        # y_test = y[test_idx]

        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        # Stratisfied split
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X,
        #     y,
        #     test_size=0.5,  # Not splitting further, just rebalancing
        #     stratify=X["Organism_Code"],
        # )

        # TODO: Check that target class ratios are the same in train / test
        y_test_list.append(y_test)
        y_train_list.append(y_train)

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

        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=X_train.columns)
        feature_importance_list.append(forest_importances.sort_values(ascending=False))

    return (
        generate_results(
            target_cols, y_test_list, y_score_list, y_pred_list, feature_importance_list
        ),
        y_test_list,
        y_train_list,
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


def run_cross_validated_random_forest(
    preprocessed_data, n_splits=5, split_strategy="random"
):
    target_cols = preprocessed_data.target_cols

    results_per_target = {}
    test_label_count_dict = {}
    train_label_count_dict = {}

    for col in target_cols:
        merged_filtered_input = preprocessed_data.merged_input
        merged_filtered_input = merged_filtered_input[merged_filtered_input[col] != 0]

        # Drop rows of Organisms that occur only once
        value_counts = merged_filtered_input["Organism_Code"].value_counts()
        rare_values = value_counts[value_counts == 1].index
        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input["Organism_Code"].isin(rare_values)
        ]

        X = merged_filtered_input[preprocessed_data.feature_cols]
        y = merged_filtered_input[col]

        stratify_col = merged_filtered_input["Organism_Code"]

        # Prepare cross-validation
        if split_strategy == "stratified":
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iterator = kf.split(X, stratify_col)
        elif split_strategy == "random":
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iterator = kf.split(X)
        elif split_strategy == "clustered":
            cluster_labels = KMeans(
                n_clusters=int((len(X) / 10)), random_state=42
            ).fit_predict(X)
            unique_clusters = np.unique(cluster_labels)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iterator = (
                (
                    np.where(np.isin(cluster_labels, unique_clusters[train_idx]))[0],
                    np.where(np.isin(cluster_labels, unique_clusters[test_idx]))[0],
                )
                for train_idx, test_idx in kf.split(unique_clusters)
            )
        else:
            raise ValueError("Invalid split_strategy")

        # Collect results from all folds
        fold_results = []

        test_label_count_list = []
        train_label_count_list = []
        for train_idx, test_idx in split_iterator:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if len(Counter(y_train)) != len(Counter(y_test)):
                print(
                    "WARNING: Skipped fold because y_train and y_test contain different classes"
                )
                continue

            test_label_count_list.append(y_test.value_counts())
            train_label_count_list.append(y_train.value_counts())

            model = RandomForestClassifier(
                n_estimators=10, class_weight="balanced", random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Map to 4-class output for y_score
            proba_full = np.zeros((y_proba.shape[0], 4))
            for idx, class_label in enumerate(model.classes_):
                proba_full[:, class_label] = y_proba[:, idx]

            importances = model.feature_importances_
            forest_importances = pd.Series(importances, index=X_train.columns)

            single_result = generate_result(
                col,
                y_test,
                proba_full,
                y_pred,
                forest_importances.sort_values(ascending=False),
            )
            fold_results.append(single_result)

        # Average metrics across folds
        results_per_target[col] = average_result_dtos(fold_results)

        test_label_dict_list = [
            s.to_dict() if isinstance(s, pd.Series) else s
            for s in test_label_count_list
        ]
        test_label_transposed = {
            key: [d[key] for d in test_label_dict_list]
            for key in test_label_dict_list[0]
        }
        test_label_count_dict[col] = {
            key: mean(values) for key, values in test_label_transposed.items()
        }
        train_label_dict_list = [
            s.to_dict() if isinstance(s, pd.Series) else s
            for s in train_label_count_list
        ]
        train_label_transposed = {
            key: [d[key] for d in train_label_dict_list]
            for key in train_label_dict_list[0]
        }
        train_label_count_dict[col] = {
            key: mean(values) for key, values in train_label_transposed.items()
        }

    return results_per_target, test_label_count_dict, train_label_count_dict


def average_result_dtos(result_dtos):
    pr_auc_dict = {}
    roc_auc_dict = {}
    accuracy_list = []
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    for dict_key in result_dtos[0].pr_auc.keys():
        pr_auc_list = []
        roc_auc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for result_dto in result_dtos:
            pr_auc_list.append(
                result_dto.pr_auc[dict_key]
            )  # TODO:     pr_auc_list.append(result_dto.pr_auc[dict_key]) KeyError: np.int64(1)
            roc_auc_list.append(result_dto.roc_auc[dict_key])
            precision_list.append(result_dto.precision[dict_key])
            recall_list.append(result_dto.recall[dict_key])
            f1_list.append(result_dto.f1[dict_key])
        pr_auc_dict[dict_key] = np.mean(pr_auc_list, axis=0)
        roc_auc_dict[dict_key] = np.mean(roc_auc_list, axis=0)
        precision_dict[dict_key] = np.mean(precision_list, axis=0)
        recall_dict[dict_key] = np.mean(recall_list, axis=0)
        f1_dict[dict_key] = np.mean(f1_list, axis=0)
    for result_dto in result_dtos:
        accuracy_list.append(result_dto.accuracy)
    return ResultDTO(
        fpr=result_dtos[
            0
        ].fpr,  # TODO: Should be mean size of S/R/I Set changes with every fold -> fpr size changes as well
        tpr=result_dtos[0].tpr,  # TODO: Should be mean
        roc_auc=roc_auc_dict,
        pr_auc=pr_auc_dict,
        accuracy=np.mean(accuracy_list),
        precision=precision_dict,
        recall=recall_dict,
        f1=f1_dict,
        y_pred=result_dtos[0].y_pred,  # TODO: Should be mean
        y_score=result_dtos[0].y_score,  # TODO: Should be mean
        feature_importance=result_dtos[0].feature_importance,  # TODO: Should be mean
    )


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


def evaluation_to_csv(results_dto, y_test_input, y_train_input):
    evaluation_df = pd.DataFrame(
        columns=[
            "Accuracy",
            "ROC_Mean",
            "PR_Mean",
            "Precision_Mean",
            "Recall_Mean",
            "F1_Mean",
            "Test_Count_S",
            "Test_Count_I",
            "Test_Count_R",
            "Train_Count_S",
            "Train_Count_I",
            "Train_Count_R",
        ]
    )
    for counter, name in enumerate(results_dto):
        result = results_dto[name]
        test_label_counts = y_test_input[name]
        train_label_counts = y_train_input[name]

        evaluation_df.loc[name] = [
            result.accuracy,
            np.nanmean(list(result.roc_auc.values())),
            np.nanmean(list(result.pr_auc.values())),
            np.nanmean(list(result.precision.values())),
            np.nanmean(list(result.recall.values())),
            np.nanmean(list(result.f1.values())),
            test_label_counts.get(1, 0),
            test_label_counts.get(2, 0),
            test_label_counts.get(3, 0),
            train_label_counts.get(1, 0),
            train_label_counts.get(2, 0),
            train_label_counts.get(3, 0),
        ]

    evaluation_df.loc["Median"] = [
        statistics.median(evaluation_df["Accuracy"].dropna()),
        statistics.median(evaluation_df["ROC_Mean"].dropna()),
        statistics.median(evaluation_df["PR_Mean"].dropna()),
        statistics.median(evaluation_df["Precision_Mean"].dropna()),
        statistics.median(evaluation_df["Recall_Mean"].dropna()),
        statistics.median(evaluation_df["F1_Mean"].dropna()),
        pd.NA,
        pd.NA,
        pd.NA,
        pd.NA,
        pd.NA,
        pd.NA,
    ]

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

    rf_results, y_test_count, y_train_count = run_cross_validated_random_forest(
        preprocessed_data_input, 5
    )

    # TODO: Split display to different class
    display_results(rf_results, False)
    evaluation_to_csv(rf_results, y_test_count, y_train_count)
