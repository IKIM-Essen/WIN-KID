# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import argparse
import os
import itertools
import random
from dataclasses import dataclass
from enum import Enum
from statistics import mean, median
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing
from constants import RESISTANCE_MAPPING
from constants import ID_COLUMN


class ModelStrategy(Enum):
    CROSS_VALIDATE = "cross_validate"
    STACKED = "stacked"
    SPLIT = "split"


class SplitStrategy(Enum):
    RANDOM = "random"
    STRATIFY = "stratify"
    CLUSTER = "cluster"


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


@dataclass
class RandomForestSettings:
    n_estimators: int
    class_weight: str
    max_depth: object
    min_samples_split: int
    min_samples_leaf: int
    max_features: str
    bootstrap: bool


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


def run_random_forest(
    X_train_input,
    y_train_input,
    X_test_input,
    y_test_input,
    target_input,
    settings_input,
):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=settings_input.n_estimators,
        class_weight=settings_input.class_weight,
        max_depth=settings_input.max_depth,
        min_samples_split=settings_input.min_samples_split,
        min_samples_leaf=settings_input.min_samples_leaf,
        max_features=settings_input.max_features,
        bootstrap=settings_input.bootstrap,
    )
    model.fit(X_train_input, y_train_input)

    y_pred = model.predict(X_test_input)
    y_proba = model.predict_proba(X_test_input)

    # Map to 4-class output for y_score
    proba_full = np.zeros((y_proba.shape[0], 4))
    for idx, class_label in enumerate(model.classes_):
        proba_full[:, class_label] = y_proba[:, idx]

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=X_train_input.columns)

    return generate_result(
        target_input,
        y_test_input,
        proba_full,
        y_pred,
        forest_importances.sort_values(ascending=False),
    )


def run_layer_one_random_forest(
    X_train_input,
    y_train_input,
    X_test_input,
    y_test_input,
    target_input,
    settings_input,
):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=settings_input.n_estimators,
        class_weight=settings_input.class_weight,
        max_depth=settings_input.max_depth,
        min_samples_split=settings_input.min_samples_split,
        min_samples_leaf=settings_input.min_samples_leaf,
        max_features=settings_input.max_features,
        bootstrap=settings_input.bootstrap,
    )

    model.fit(X_train_input, y_train_input)

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=X_train_input.columns)

    y_pred_test = model.predict(X_test_input)
    y_proba_test = model.predict_proba(X_test_input)

    # Map to 4-class output for y_score
    proba_full_test = np.zeros((y_proba_test.shape[0], 4))
    for idx, class_label in enumerate(model.classes_):
        proba_full_test[:, class_label] = y_proba_test[:, idx]

    y_proba_train = model.predict_proba(X_train_input)

    proba_full_train = np.zeros((y_proba_train.shape[0], 4))
    for idx, class_label in enumerate(model.classes_):
        proba_full_train[:, class_label] = y_proba_train[:, idx]

    column_names = [f"{res}_{target_input}" for res in RESISTANCE_MAPPING.keys()]
    proba_full_train_df = pd.DataFrame(proba_full_train, columns=column_names)
    proba_full_test_df = pd.DataFrame(proba_full_test, columns=column_names)

    result_test = generate_result(
        target_input,
        y_test_input,
        proba_full_test,
        y_pred_test,
        forest_importances.sort_values(ascending=False),
    )
    return result_test, proba_full_train_df, proba_full_test_df


def run_splitted_random_forest(
    preprocessed_data, test_size, split_strategy, rf_settings
):

    y_test_count, y_train_count, result_dic = ({}, {}, {})
    y_test, y_train, X_test, X_train = ([], [], [], [])

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

        if split_strategy == SplitStrategy.STRATIFY:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,  # Not splitting further, just rebalancing
                stratify=X["Organism_Code"],
            )
        elif split_strategy == SplitStrategy.RANDOM:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        elif split_strategy == SplitStrategy.CLUSTER:
            cluster_labels = KMeans(
                n_clusters=int((len(merged_filtered_input) / 10)), random_state=42
            ).fit_predict(X)
            unique_clusters = np.unique(cluster_labels)
            train_clusters, test_clusters = train_test_split(
                unique_clusters, test_size=test_size, random_state=42
            )
            train_idx = np.isin(cluster_labels, train_clusters)
            test_idx = np.isin(cluster_labels, test_clusters)
            X_train, X_test, y_train, y_test = (
                X[train_idx],
                X[test_idx],
                y[train_idx],
                y[test_idx],
            )

        y_test_count[col] = Counter(y_test)
        y_train_count[col] = Counter(y_train)

        sinlge_result = run_random_forest(
            X_train, y_train, X_test, y_test, col, rf_settings
        )
        result_dic[col] = sinlge_result

    return (
        result_dic,
        y_test_count,
        y_train_count,
    )


def run_stacked_random_forest(
    preprocessed_data, test_size, split_strategy, rf_settings
):

    # Prepare first layer data
    y_test_count, y_train_count, result_dic = ({}, {}, {})

    merged_filtered_input = preprocessed_data.merged_input
    # Drop rows of Organisms that occur only once
    value_counts = preprocessed_data.merged_input["Organism_Code"].value_counts()
    rare_values = value_counts[value_counts == 1].index
    merged_filtered_input = merged_filtered_input[
        ~merged_filtered_input["Organism_Code"].isin(rare_values)
    ]

    feature_cols_with_id = preprocessed_data.feature_cols
    feature_cols_with_id.append(ID_COLUMN)
    X = preprocessed_data.merged_input[feature_cols_with_id]
    y = preprocessed_data.merged_input[preprocessed_data.target_cols]

    # SPLITTING

    y_test, y_train, X_test, X_train = split_sets_for_stacked(
        test_size, split_strategy, X, y
    )

    proba_train_joined = pd.DataFrame([])
    proba_test_joined = pd.DataFrame([])

    # FIRST LAYER

    proba_train_joined, proba_test_joined = run_first_layer(
        preprocessed_data,
        rf_settings,
        y_test,
        y_train,
        X_test,
        X_train,
        proba_train_joined,
        y_train_count,
        y_test_count,
    )

    y_proba_train, X_proba_train, y_proba_test, X_proba_test = (
        prepare_second_layer_data(
            preprocessed_data,
            merged_filtered_input,
            proba_train_joined,
            proba_test_joined,
        )
    )

    # TODO: Try zero instead of NaN for unpredicted ABs -> Does not seem to make a big difference
    # print(X_proba_test)
    # X_proba_train = X_proba_train.fillna(0.0)
    # X_proba_test = X_proba_test.fillna(0.0)
    # print(X_proba_test)

    # SECOND LAYER
    for target in preprocessed_data.target_cols:
        second_layer_result = run_random_forest(
            X_proba_train,
            y_proba_train[target],
            X_proba_test,
            y_proba_test[target],
            target,
            rf_settings,
        )
        result_dic[target] = second_layer_result

    return (
        result_dic,
        y_test_count,
        y_train_count,
    )


def prepare_second_layer_data(
    preprocessed_data, merged_filtered_input, proba_train_joined, proba_test_joined
):
    y_proba_train = merged_filtered_input[
        merged_filtered_input[ID_COLUMN].isin(proba_train_joined[ID_COLUMN]).copy()
    ]
    proba_train_joined = proba_train_joined.sort_values(by=[ID_COLUMN])
    y_proba_train = y_proba_train.sort_values(by=[ID_COLUMN])
    y_proba_train = y_proba_train[preprocessed_data.target_cols]
    X_proba_train = proba_train_joined.drop(ID_COLUMN, axis=1)

    y_proba_test = merged_filtered_input[
        merged_filtered_input[ID_COLUMN].isin(proba_test_joined[ID_COLUMN]).copy()
    ]
    proba_test_joined = proba_test_joined.sort_values(by=[ID_COLUMN])
    y_proba_test = y_proba_test.sort_values(by=[ID_COLUMN])
    y_proba_test = y_proba_test[preprocessed_data.target_cols]
    X_proba_test = proba_test_joined.drop(ID_COLUMN, axis=1)
    return y_proba_train, X_proba_train, y_proba_test, X_proba_test


def run_first_layer(
    preprocessed_data,
    rf_settings,
    y_test,
    y_train,
    X_test,
    X_train,
    proba_train_joined,
    y_train_count,
    y_test_count,
):
    for target in preprocessed_data.target_cols:
        y_train_target = y_train[target]
        y_test_target = y_test[target]

        mask_train = y_train_target != 0
        y_train_target = y_train_target[mask_train]
        X_train_target = X_train[mask_train]
        mask_test = y_test_target != 0
        y_test_target = y_test_target[mask_test]
        X_test_target = X_test[mask_test]

        X_train_target_id = X_train_target[ID_COLUMN]
        X_train_target = X_train_target.drop(ID_COLUMN, axis=1)
        X_test_target_id = X_test_target[ID_COLUMN]
        X_test_target = X_test_target.drop(ID_COLUMN, axis=1)

        y_train_count[target] = Counter(y_train_target)
        y_test_count[target] = Counter(y_test_target)

        _, proba_train_target, proba_test_target = run_layer_one_random_forest(
            X_train_target,
            y_train_target,
            X_test_target,
            y_test_target,
            target,
            rf_settings,
        )

        # Drop unused NaN column
        proba_train_target = proba_train_target.drop(
            proba_train_target.columns[0], axis=1
        )
        proba_test_target = proba_test_target.drop(proba_test_target.columns[0], axis=1)

        # Add ID back again
        proba_train_target: pd.DataFrame
        proba_train_target[ID_COLUMN] = X_train_target_id.reset_index(drop=True)
        proba_test_target: pd.DataFrame
        proba_test_target[ID_COLUMN] = X_test_target_id.reset_index(drop=True)

        if len(proba_train_joined) == 0:
            proba_train_joined = proba_train_target
            proba_test_joined = proba_test_target
        else:
            proba_train_joined = pd.merge(
                proba_train_joined, proba_train_target, on=ID_COLUMN, how="outer"
            )
            proba_test_joined = pd.merge(
                proba_test_joined, proba_test_target, on=ID_COLUMN, how="outer"
            )

    return proba_train_joined, proba_test_joined


def split_sets_for_stacked(test_size, split_strategy, X, y):
    if split_strategy == SplitStrategy.STRATIFY:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,  # Not splitting further, just rebalancing
            stratify=X["Organism_Code"],
        )
    elif split_strategy == SplitStrategy.RANDOM:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    elif split_strategy == SplitStrategy.CLUSTER:
        # ID_COLUMN shall not be used to cluster
        X_clustering = X.drop(columns=[ID_COLUMN])  # Or multiple columns

        cluster_labels = KMeans(
            n_clusters=int(len(X) / 10), random_state=42
        ).fit_predict(X_clustering)

        unique_clusters = np.unique(cluster_labels)
        train_clusters, test_clusters = train_test_split(
            unique_clusters, test_size=test_size, random_state=42
        )
        train_idx = np.isin(cluster_labels, train_clusters)
        test_idx = np.isin(cluster_labels, test_clusters)

        # Now use full X (with ID_COLUMN) for model input/output
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        raise ValueError("Split strategy !" + str(split_strategy) + "! is invalid")

    return y_test, y_train, X_test, X_train


def load_dataset_paths(path_file):
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Settings file '{path_file}' not found.")

    path_df = pd.read_csv(path_file)

    if not {"DataSetName", "PathToCsv", "PathToGff"}.issubset(path_df.columns):
        raise ValueError(
            "Settings file must contain columns: DataSetName, PathToCsv, PathToGff"
        )

    return path_df


def get_split_iterator(X, stratify_col, strategy, n_splits):
    if strategy == SplitStrategy.STRATIFY:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            X, stratify_col
        )
    elif strategy == SplitStrategy.RANDOM:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X)
    elif strategy == SplitStrategy.CLUSTER:
        cluster_labels = KMeans(
            n_clusters=int(len(X) / 10), random_state=42
        ).fit_predict(X)
        unique_clusters = np.unique(cluster_labels)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return (
            (
                np.where(np.isin(cluster_labels, unique_clusters[train_idx]))[0],
                np.where(np.isin(cluster_labels, unique_clusters[test_idx]))[0],
            )
            for train_idx, test_idx in kf.split(unique_clusters)
        )
    else:
        raise ValueError("Invalid split_strategy")


def compute_label_distribution(label_counts_list):
    dict_list = [
        s.to_dict() if isinstance(s, pd.Series) else s for s in label_counts_list
    ]
    transposed = {key: [d[key] for d in dict_list] for key in dict_list[0]}
    return {key: mean(values) for key, values in transposed.items()}


def run_cross_validated_random_forest(
    preprocessed_data, n_splits, split_strategy, rf_settings_cv
):
    results_per_target = {}
    test_label_count_dict = {}
    train_label_count_dict = {}

    for col in preprocessed_data.target_cols:

        # Remove rows with label 0 (unlabeled)
        df = preprocessed_data.merged_input[preprocessed_data.merged_input[col] != 0]
        # Remove rare Organism_Code values (only occur once)
        organism_counts = df["Organism_Code"].value_counts()
        common_organisms = organism_counts[organism_counts > 1].index
        df = df[df["Organism_Code"].isin(common_organisms)]

        X = df[preprocessed_data.feature_cols]
        y = df[col]
        stratify_col = df["Organism_Code"]

        split_iterator = get_split_iterator(X, stratify_col, split_strategy, n_splits)

        fold_results = []
        test_label_counts = []
        train_label_counts = []

        for train_idx, test_idx in split_iterator:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if set(y_train.unique()) != set(y_test.unique()):
                print(
                    f"Skipped fold for {col}: y_train and y_test have different classes."
                )
                continue

            test_label_counts.append(y_test.value_counts())
            train_label_counts.append(y_train.value_counts())

            result = run_random_forest(
                X_train, y_train, X_test, y_test, col, rf_settings_cv
            )
            fold_results.append(result)

        if fold_results:
            results_per_target[col] = average_result_dtos(fold_results)
            test_label_count_dict[col] = compute_label_distribution(test_label_counts)
            train_label_count_dict[col] = compute_label_distribution(train_label_counts)

    return results_per_target, test_label_count_dict, train_label_count_dict


def average_result_dtos(result_dtos):
    pr_auc_dict, roc_auc_dict, precision_dict, recall_dict, f1_dict = (
        {},
        {},
        {},
        {},
        {},
    )
    accuracy_list = []
    for dict_key in result_dtos[0].pr_auc.keys():
        pr_auc_list, roc_auc_list, precision_list, recall_list, f1_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for result_dto in result_dtos:
            pr_auc_list.append(result_dto.pr_auc[dict_key])
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
        None,  # Should be mean size. S/R/I Set changes with every fold -> fpr size changes as well
        None,
        roc_auc=roc_auc_dict,
        pr_auc=pr_auc_dict,
        accuracy=np.mean(accuracy_list),
        precision=precision_dict,
        recall=recall_dict,
        f1=f1_dict,
        y_pred=None,
        y_score=None,
        feature_importance=None,
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
    for _, name in enumerate(results_dto):
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
        median(evaluation_df["Accuracy"].dropna()),
        median(evaluation_df["ROC_Mean"].dropna()),
        median(evaluation_df["PR_Mean"].dropna()),
        median(evaluation_df["Precision_Mean"].dropna()),
        median(evaluation_df["Recall_Mean"].dropna()),
        median(evaluation_df["F1_Mean"].dropna()),
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


def tune_hyperparameter(preprocessed_data, number_of_folds):
    param_grid = {
        "n_estimators": [10, 50, 100, 200, 500],
        "max_depth": [None, 10, 20, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3, None],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True, False],
        "split_strategy": list(SplitStrategy),
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    combinations_subsample = random.sample(combinations, min(3, len(combinations)))

    metrics = {
        "Accuracy": [],
        "ROC_AUC": [],
        "PR_AUC": [],
        "Precision": [],
        "Recall": [],
        "f1": [],
    }

    combinations_subsample_df = pd.DataFrame(combinations_subsample)
    counter = 1
    for combo in combinations_subsample:
        print(f"{counter} of {len(combinations_subsample)} subsampled combinations")
        counter = counter + 1

        max_depth_value = combo["max_depth"]
        if combo["max_depth"] != combo["max_depth"]:
            max_depth_value = None

        rf_settings = RandomForestSettings(
            n_estimators=combo["n_estimators"],
            class_weight=combo["class_weight"],
            max_depth=max_depth_value,
            min_samples_split=combo["min_samples_split"],
            min_samples_leaf=combo["min_samples_leaf"],
            max_features=combo["max_features"],
            bootstrap=combo["bootstrap"],
        )

        rf_cv_results, _, _ = run_cross_validated_random_forest(
            preprocessed_data,
            number_of_folds,
            SplitStrategy(combo["split_strategy"].value),
            rf_settings,
        )

        accs, rocs, prs, precs, recs, f1s = [], [], [], [], [], []
        for result in rf_cv_results.values():
            accs.append(result.accuracy)
            rocs.append(np.nanmean(list(result.roc_auc.values())))
            prs.append(np.nanmean(list(result.pr_auc.values())))
            precs.append(np.nanmean(list(result.precision.values())))
            recs.append(np.nanmean(list(result.recall.values())))
            f1s.append(np.nanmean(list(result.f1.values())))

        metrics["Accuracy"].append(median(accs))
        metrics["ROC_AUC"].append(median(rocs))
        metrics["PR_AUC"].append(median(prs))
        metrics["Precision"].append(median(precs))
        metrics["Recall"].append(median(recs))
        metrics["f1"].append(median(f1s))

    for key, values in metrics.items():
        combinations_subsample_df[key] = values

    combinations_subsample_df.to_csv(
        "Evaluation/Hyperparameter.csv", index=True, header=True
    )


if __name__ == "__main__":
    TUNE_HYPERPARAMETER = False
    MODEL_STRATEGY = ModelStrategy.STACKED

    NUMBER_OF_FOLDS = 5
    TEST_SIZE = 0.3

    SPLIT_STRATEGY = SplitStrategy.STRATIFY
    rf_settings_input = RandomForestSettings(
        n_estimators=100,
        class_weight="balanced_subsample",
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.3,
        bootstrap=True,
    )

    parser = argparse.ArgumentParser(
        description="Run RF on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")
    args = parser.parse_args()

    dataset_list = load_dataset_paths(args.path_file)

    data_loader = preprocessing.DataLoader()
    preprocessed_data_input = data_loader.get_preprocessed_data(dataset_list)

    if TUNE_HYPERPARAMETER is True:
        tune_hyperparameter(preprocessed_data_input, NUMBER_OF_FOLDS)

    else:
        if MODEL_STRATEGY is ModelStrategy.CROSS_VALIDATE:
            # display not possible with CV.
            # S/R/I Set changes with every fold -> fpr size changes as well
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
            ) = run_cross_validated_random_forest(
                preprocessed_data_input,
                NUMBER_OF_FOLDS,
                SPLIT_STRATEGY,
                rf_settings_input,
            )
        elif MODEL_STRATEGY is ModelStrategy.STACKED:
            # TODO: Cross validate
            # TODO: Tune Hyperparameters for each model
            # TODO: Add ID_COLUMN assertion
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
            ) = run_stacked_random_forest(
                preprocessed_data_input, TEST_SIZE, SPLIT_STRATEGY, rf_settings_input
            )

        elif MODEL_STRATEGY is ModelStrategy.SPLIT:

            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
            ) = run_splitted_random_forest(
                preprocessed_data_input, TEST_SIZE, SPLIT_STRATEGY, rf_settings_input
            )

            display_results(rf_results, False)

        else:
            raise ValueError("Model strategy !" + str(MODEL_STRATEGY) + "! is invalid")

        evaluation_to_csv(rf_results, y_test_count_result, y_train_count_result)
