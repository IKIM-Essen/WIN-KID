# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import time
import argparse
import os
import itertools
import random
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from statistics import mean, median
from datetime import datetime
from collections import Counter, defaultdict
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
import cloudpickle
import config
from split_strategies import SplitStrategy
from constants import RESISTANCE_MAPPING
from constants import ID_COLUMN
from constants import ORGANISM_COLUMN
from constants import MODEL_FOLDER
from execution_modes import ExecutionMode


class ModelStrategy(Enum):
    CROSS_VALIDATE = "cross_validate"
    STACKED = "stacked"
    SPLIT = "split"


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
    vme: dict
    me: dict
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
    vme = {}
    me = {}

    if config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
        fpr, tpr, roc_auc, accuracy, precision, recall, f1, vme, me = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
    else:
        for label_class in unique_classes:
            y_test_binarized = (y_test_col == label_class).astype(int)
            y_pred_binarized = (y_pred_col == label_class).astype(int)

            if label_class >= y_score_col.shape[1]:
                print(
                    f"Skipping class {label_class} for {target_col}, not in predictions"
                )
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
        vme = calc_very_major_errors(y_test_col, y_pred_col)
        me = calc_major_errors(y_test_col, y_pred_col)

    return ResultDTO(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        vme=vme,
        me=me,
        y_pred=y_pred_col,
        y_score=y_score_col,
        feature_importance=feat_import,
    )


def calc_very_major_errors(y_test_col, y_pred_col):
    mask = (y_test_col == 3) & np.isin(y_pred_col, [1, 2])
    count = mask.sum()
    percentage_vme = count / len(y_test_col) * 100
    return percentage_vme


def calc_major_errors(y_test_col, y_pred_col):
    mask = (y_pred_col == 3) & np.isin(y_test_col, [1, 2])
    count = mask.sum()
    percentage_all = count / len(y_test_col) * 100
    return percentage_all


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

    # Sort features
    X_train_input = X_train_input.reindex(sorted(X_train_input.columns), axis=1)
    X_test_input = X_test_input.reindex(sorted(X_test_input.columns), axis=1)

    model_path = MODEL_FOLDER + "second_layer/" + target_input + ".pkl"
    if (
        config.EXECUTION_MODE == ExecutionMode.TRAIN_TEST
        or config.EXECUTION_MODE == ExecutionMode.TUNE_HYPERPARAMETER
    ):
        validate_target_values(y_train_input, y_test_input, target_input)
        model.fit(X_train_input, y_train_input)

    elif config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
        validate_target_values(y_train_input, y_test_input, target_input)
        model.fit(X_train_input, y_train_input)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)

    elif config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)

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
    X_val_input,
    y_val_input,
    X_test_input,
    target_input,
    settings_input,
    run_number,
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
    model_path = (
        MODEL_FOLDER
        + "first_layer/"
        + str(run_number)
        + "_run/"
        + target_input
        + ".pkl"
    )

    # Sort features
    X_train_input = X_train_input.reindex(sorted(X_train_input.columns), axis=1)
    X_val_input = X_val_input.reindex(sorted(X_val_input.columns), axis=1)
    X_test_input = X_test_input.reindex(sorted(X_test_input.columns), axis=1)

    if config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
        validate_target_values(y_train_input, y_val_input, target_input)
        model.fit(X_train_input, y_train_input)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)

    elif config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
        X_test_input = X_test_input.drop(ID_COLUMN, axis=1)
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)

    elif config.EXECUTION_MODE == ExecutionMode.TRAIN_TEST:
        validate_target_values(y_train_input, y_val_input, target_input)
        model.fit(X_train_input, y_train_input)

    y_proba_val = model.predict_proba(X_val_input)
    y_proba_test = model.predict_proba(X_test_input)

    # Map to 4-class output for y_score
    proba_full_val = np.zeros((y_proba_val.shape[0], 4))
    for idx, class_label in enumerate(model.classes_):
        proba_full_val[:, class_label] = y_proba_val[:, idx]
    # Map to 4-class output for y_score
    proba_full_test = np.zeros((y_proba_test.shape[0], 4))
    for idx, class_label in enumerate(model.classes_):
        proba_full_test[:, class_label] = y_proba_test[:, idx]

    column_names = [f"{res}_{target_input}" for res in RESISTANCE_MAPPING.keys()]
    proba_full_val_df = pd.DataFrame(proba_full_val, columns=column_names)
    proba_full_test_df = pd.DataFrame(proba_full_test, columns=column_names)

    return proba_full_val_df, proba_full_test_df


def run_splitted_random_forest(preprocessed_data, rf_settings):

    y_test_count, y_train_count, result_dic = ({}, {}, {})
    y_test, y_train, X_test, X_train = ([], [], [], [])
    org_res = {}

    for col in preprocessed_data.target_cols:
        merged_filtered_input = preprocessed_data.merged_input
        merged_filtered_input = merged_filtered_input[merged_filtered_input[col] != 0]

        # Drop rows of Organisms that occur only once
        value_counts = merged_filtered_input[ORGANISM_COLUMN].value_counts()
        rare_values = value_counts[value_counts == 1].index
        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input[ORGANISM_COLUMN].isin(rare_values)
        ]

        X = merged_filtered_input[preprocessed_data.feature_cols]
        y = merged_filtered_input[col]

        if config.SPLIT_STRATEGY.name == SplitStrategy.STRATIFY.name:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.TEST_SIZE,  # Not splitting further, just rebalancing
                stratify=X[ORGANISM_COLUMN],
            )
        elif config.SPLIT_STRATEGY.name == SplitStrategy.RANDOM.name:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=42
            )
        elif config.SPLIT_STRATEGY.name == SplitStrategy.CLUSTER.name:
            cluster_labels = KMeans(
                n_clusters=int((len(merged_filtered_input) / 10)), random_state=42
            ).fit_predict(X)
            unique_clusters = np.unique(cluster_labels)
            train_clusters, test_clusters = train_test_split(
                unique_clusters, test_size=config.TEST_SIZE, random_state=42
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

        organism_test = merged_filtered_input.loc[y_test.index, ORGANISM_COLUMN]
        org_res[col] = evaluate_per_organism(
            sinlge_result.y_pred, y_test, organism_test, sinlge_result.y_score, col
        )

    return (
        [result_dic],
        [y_test_count],
        [y_train_count],
        org_res,
    )


def run_stacked_random_forest(
    preprocessed_data,
    rf_settings_first_layer,
    rf_settings_second_layer,
    cross_validate=False,
):

    # Prepare first layer data

    merged_filtered_input = preprocessed_data.merged_input

    if config.EXECUTION_MODE != ExecutionMode.PREDICT_ON_SAVED:
        # Drop rows of Organisms that occur only once
        value_counts = preprocessed_data.merged_input[ORGANISM_COLUMN].value_counts()
        rare_values = value_counts[value_counts == 1].index
        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input[ORGANISM_COLUMN].isin(rare_values)
        ]

    feature_cols_with_id = preprocessed_data.feature_cols + [ID_COLUMN]
    X = merged_filtered_input[feature_cols_with_id]
    y = merged_filtered_input[preprocessed_data.target_cols]

    if cross_validate:

        if config.EXECUTION_MODE != ExecutionMode.TRAIN_TEST:
            raise ValueError("--mode shall be TRAIN_TEST for cross validation")

        skf = StratifiedKFold(
            n_splits=config.NUMBER_OF_FOLDS, shuffle=True, random_state=42
        )
        stratify_col = preprocessed_data.merged_input[ORGANISM_COLUMN]

        cv_results = []
        test_label_counts = []
        train_label_counts = []
        org_res = {}
        fold_per_organism_results = defaultdict(list)
        for train_idx, test_idx in skf.split(X, stratify_col):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            (
                result_dic,
                y_test_count,
                y_train_count,
                fold_per_organism_result,
            ) = compute_stacked_random_forest(
                preprocessed_data,
                rf_settings_first_layer,
                rf_settings_second_layer,
                merged_filtered_input,
                X_train,
                X_test,
                y_train,
                y_test,
            )

            cv_results.append(result_dic)
            test_label_counts.append(y_test_count)
            train_label_counts.append(y_train_count)

            for target, per_target_result in fold_per_organism_result.items():
                fold_per_organism_results[target].append(per_target_result)
        for target, per_target_results in fold_per_organism_results.items():
            org_res[target] = average_per_organism_results(per_target_results)

        return (
            cv_results,
            test_label_counts,
            train_label_counts,
            org_res,
        )
    else:
        # SPLITTING
        if config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
            print(
                "WARNING in "
                + config.EXECUTION_MODE.value
                + " all samples are used for test"
            )
            y_test = y
            y_train = y
            X_test = X
            X_train = X
        else:
            y_test, y_train, X_test, X_train = split_sets_for_stacked(X, y)

        # Train with all sample if saved
        if config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
            print(
                "WARNING in "
                + config.EXECUTION_MODE.value
                + " all samples are used for training"
            )
            y_train = pd.concat([y_train, y_test], ignore_index=True)
            X_train = pd.concat([X_train, X_test], ignore_index=True)

        (
            result_dic,
            y_test_count,
            y_train_count,
            org_res,
        ) = compute_stacked_random_forest(
            preprocessed_data,
            rf_settings_first_layer,
            rf_settings_second_layer,
            merged_filtered_input,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        if config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open((MODEL_FOLDER + "run_info.txt"), "w", encoding="utf-8") as f:
                f.write(f"Run executed at: {now}\n")

        return (
            [result_dic],
            [y_test_count],
            [y_train_count],
            org_res,
        )


def filter_merged_input(preprocessed_data, min_sample_number):
    merged_filtered_input = preprocessed_data.merged_input

    # Remove classes and ABs that rarely occur
    for col_name in preprocessed_data.target_cols:
        value_counts = merged_filtered_input[col_name].value_counts()
        low_freq_values = value_counts[value_counts < min_sample_number].index

        if len(list(low_freq_values)) > 0:
            print(f"Removed values for column '{col_name}': {list(low_freq_values)}")

        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input[col_name].isin(low_freq_values)
        ]
        updated_value_counts = merged_filtered_input[col_name].value_counts()

        if len(updated_value_counts) <= 2:
            merged_filtered_input = merged_filtered_input.drop(col_name, axis=1)
            print(col_name + " removed because only one class is left after filtering")
            preprocessed_data.target_cols = preprocessed_data.target_cols.difference(
                [col_name]
            )
    # Remove full NaN rows
    mask = (merged_filtered_input[preprocessed_data.target_cols] != 0).any(axis=1)
    merged_filtered_input = merged_filtered_input[mask]

    preprocessed_data.merged_input = merged_filtered_input
    print(
        "Number of samples after sample number filtering: "
        + str(len(preprocessed_data.merged_input))
    )
    print(preprocessed_data.merged_input[ORGANISM_COLUMN].value_counts())
    os.makedirs("Evaluation", exist_ok=True)
    preprocessed_data.merged_input[ID_COLUMN].to_csv(
        "Evaluation/samples_used.csv", index=False
    )

    return preprocessed_data


def evaluate_per_organism(y_pred, y_true, organism_codes, y_score, target_name):
    results = {}

    df = pd.DataFrame({"organism": organism_codes, "y_true": y_true, "y_pred": y_pred})

    for i in range(y_score.shape[1]):
        df[f"proba_class_{i}"] = y_score[:, i]

    for org_code, group in df.groupby("organism"):
        y_t = group["y_true"].values
        y_p = group["y_pred"].values

        proba_cols = [f"proba_class_{i}" for i in range(y_score.shape[1])]
        y_s = group[proba_cols].values

        # Skip if only 1 class is present (not valid for AUC)
        if len(np.unique(y_t)) < 2:
            print(
                f"[WARN] Skipping evaluation per organism for organism {org_code} due to single class in target '{target_name}'."
            )

            roc_auc = pr_auc = np.nan
        else:
            result = generate_result("per_organism", y_t, y_s, y_p, feat_import=None)

            # Use mean of per-class metrics as representative score
            roc_auc = np.nanmean(list(result.roc_auc.values()))
            pr_auc = np.nanmean(list(result.pr_auc.values()))
            precision = np.nanmean(list(result.precision.values()))
            recall = np.nanmean(list(result.recall.values()))
            f1 = np.nanmean(list(result.f1.values()))
            acc = result.accuracy

            results[org_code] = {
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

    return results


def compute_stacked_random_forest(
    preprocessed_data,
    rf_settings_first_layer,
    rf_settings_second_layer,
    merged_filtered_input,
    X_train,
    X_test,
    y_train,
    y_test,
):
    y_test_count, y_train_count, result_dic = ({}, {}, {})

    proba_train_joined = pd.DataFrame([])
    proba_test_joined = pd.DataFrame([])

    # FIRST LAYER
    proba_train_joined, proba_test_joined = run_first_layer(
        preprocessed_data,
        rf_settings_first_layer,
        y_test,
        y_train,
        X_test,
        X_train,
        proba_train_joined,
        y_train_count,
        y_test_count,
    )

    (
        y_proba_train,
        X_proba_train,
        y_proba_test,
        X_proba_test,
        organism_test,
    ) = prepare_second_layer_data(
        preprocessed_data,
        merged_filtered_input,
        proba_train_joined,
        proba_test_joined,
    )

    # SECOND LAYER
    org_res = {}

    for target in preprocessed_data.target_cols:

        # Filter out NA values
        train_mask = y_proba_train[target] != 0
        X_train_filtered = X_proba_train[train_mask]
        y_train_filtered = y_proba_train[target][train_mask]
        test_mask = y_proba_test[target] != 0
        X_test_filtered = X_proba_test[test_mask]
        y_test_filtered = y_proba_test[target][test_mask]
        organism_test_filtered = organism_test[test_mask]

        if config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
            X_train_filtered = X_proba_train
            y_train_filtered = y_proba_train[target]
            X_test_filtered = X_proba_test
            y_test_filtered = y_proba_test[target]

        second_layer_result = run_random_forest(
            X_train_filtered,
            y_train_filtered,
            X_test_filtered,
            y_test_filtered,
            target,
            rf_settings_second_layer,
        )
        result_dic[target] = second_layer_result

        y_pred = second_layer_result.y_pred

        per_organism_perf = evaluate_per_organism(
            y_pred,
            y_test_filtered,
            organism_test_filtered,
            second_layer_result.y_score,
            target,
        )

        # Mapping organism name to string
        organism_mapping = preprocessed_data.organism_mapping

        org_res[target] = {
            organism_mapping.get(org_code, f"Unknown ({org_code})"): metrics
            for org_code, metrics in per_organism_perf.items()
        }

    return result_dic, y_test_count, y_train_count, org_res


def validate_target_values(y_train, y_test, target_name):
    if (y_train == 0).any():
        raise ValueError(f"❌ 0 value found in y_train for target '{target_name}'")
    if (y_test == 0).any():
        raise ValueError(f"❌ 0 value found in y_test for target '{target_name}'")
    if y_train.isna().any():
        raise ValueError(f"❌ NaN value found in y_train for target '{target_name}'")
    if y_test.isna().any():
        raise ValueError(f"❌ NaN value found in y_test for target '{target_name}'")


def prepare_second_layer_data(
    preprocessed_data, merged_filtered_input, proba_train_joined, proba_test_joined
):
    column_list = list(preprocessed_data.target_cols.copy())
    column_list.append(ID_COLUMN)

    y_proba_train = pd.merge(
        proba_train_joined[[ID_COLUMN]],
        merged_filtered_input[column_list],
        on=ID_COLUMN,
        how="left",
    )[preprocessed_data.target_cols]

    X_proba_train = proba_train_joined.merge(
        merged_filtered_input[[ID_COLUMN, ORGANISM_COLUMN]],
        on=ID_COLUMN,
        how="left",  # or 'inner', depending on what you want
    )
    X_proba_train = X_proba_train.drop(columns=[ID_COLUMN])

    y_proba_test = pd.merge(
        proba_test_joined[[ID_COLUMN]],
        merged_filtered_input[column_list],
        on=ID_COLUMN,
        how="left",
    )[preprocessed_data.target_cols]

    X_proba_test = proba_test_joined.merge(
        merged_filtered_input[[ID_COLUMN, ORGANISM_COLUMN]],
        on=ID_COLUMN,
        how="left",  # or 'inner', depending on what you want
    )
    X_proba_test = X_proba_test.drop(columns=[ID_COLUMN])

    organism_test = pd.merge(
        proba_test_joined[[ID_COLUMN]],
        merged_filtered_input[[ID_COLUMN, ORGANISM_COLUMN]],
        on=ID_COLUMN,
        how="left",
    )[ORGANISM_COLUMN].astype(int)

    return y_proba_train, X_proba_train, y_proba_test, X_proba_test, organism_test


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
    n_splits = 5
    for target in preprocessed_data.target_cols:
        y_train_target = y_train[target]
        y_test_target = y_test[target]

        mask_train = y_train_target != 0
        y_train_target = y_train_target[mask_train]
        X_train_target = X_train[mask_train]
        mask_test = y_test_target != 0
        y_test_target = y_test_target[mask_test]
        X_test_target = X_test[mask_test]

        X_test_target_id = X_test_target[ID_COLUMN]
        X_test_target = X_test_target.drop(ID_COLUMN, axis=1)

        y_train_count[target] = Counter(y_train_target)
        y_test_count[target] = Counter(y_test_target)
        if config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
            X_train_target = X_test
            X_test_target = X_test
            X_test_target_id = X_test_target[ID_COLUMN]
            y_train_target = y_train[target]
        proba_train_target, proba_test_list = compute_oof_predictions(
            rf_settings, n_splits, target, y_train_target, X_train_target, X_test_target
        )
        stacked_test = pd.concat(proba_test_list).groupby(level=0)
        proba_test_target = stacked_test.median()
        # Drop unused NaN column
        proba_train_target = proba_train_target.drop(
            proba_train_target.columns[0], axis=1
        )
        proba_test_target = proba_test_target.drop(proba_test_target.columns[0], axis=1)

        # Add ID back again
        proba_test_target: pd.DataFrame
        proba_test_target[ID_COLUMN] = X_test_target_id.reset_index(drop=True)

        if proba_train_joined.empty:
            proba_train_joined = proba_train_target
            proba_test_joined = proba_test_target
        else:
            proba_train_joined = pd.concat(
                [
                    proba_train_joined.set_index(ID_COLUMN),
                    proba_train_target.set_index(ID_COLUMN),
                ],
                axis=1,
            ).reset_index()
            proba_test_joined = pd.concat(
                [
                    proba_test_joined.set_index(ID_COLUMN),
                    proba_test_target.set_index(ID_COLUMN),
                ],
                axis=1,
            ).reset_index()

    return proba_train_joined, proba_test_joined


def compute_oof_predictions(
    rf_settings, n_splits, target, y_train_target, X_train_target, X_test_target
):
    proba_train_target = pd.DataFrame([])
    proba_test_list = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    run_counter = 0
    # Add additional train data if necessary for splitting. Train data has no effect on prediction.
    if (
        config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED
        and len(X_train_target) < 5
    ):
        X_first_row_repeated = pd.concat(
            [X_train_target.iloc[[0]]] * 4, ignore_index=True
        )
        X_train_target = pd.concat(
            [X_first_row_repeated, X_train_target], ignore_index=True
        )
        y_first_row_repeated = pd.concat(
            [y_train_target.iloc[[0]]] * 4, ignore_index=True
        )
        y_train_target = pd.concat(
            [y_first_row_repeated, y_train_target], ignore_index=True
        )
    for train_idx, valid_idx in skf.split(X_train_target, y_train_target):
        run_counter = run_counter + 1
        X_tr, X_val = X_train_target.iloc[train_idx], X_train_target.iloc[valid_idx]
        y_tr, y_val = y_train_target.iloc[train_idx], y_train_target.iloc[valid_idx]
        X_val_target_id = X_val[ID_COLUMN]
        X_val = X_val.drop(ID_COLUMN, axis=1)
        X_tr = X_tr.drop(ID_COLUMN, axis=1)

        (fold_val_pred, fold_test_pred) = run_layer_one_random_forest(
            X_tr, y_tr, X_val, y_val, X_test_target, target, rf_settings, run_counter
        )

        fold_val_pred[ID_COLUMN] = X_val_target_id.reset_index(drop=True)

        if proba_train_target.empty:
            proba_train_target = fold_val_pred
        else:
            proba_train_target = pd.concat(
                [proba_train_target, fold_val_pred], ignore_index=True
            )
        proba_test_list.append(fold_test_pred)
    return proba_train_target, proba_test_list


def split_sets_for_stacked(X, y):
    if config.SPLIT_STRATEGY.name == SplitStrategy.STRATIFY.name:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,  # Not splitting further, just rebalancing
            stratify=X[ORGANISM_COLUMN],
        )
    elif config.SPLIT_STRATEGY.name == SplitStrategy.RANDOM.name:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=42
        )
    elif config.SPLIT_STRATEGY.name == SplitStrategy.CLUSTER.name:
        # ID_COLUMN shall not be used to cluster
        X_clustering = X.drop(columns=[ID_COLUMN])  # Or multiple columns

        cluster_labels = KMeans(
            n_clusters=int(len(X) / 10), random_state=42
        ).fit_predict(X_clustering)

        unique_clusters = np.unique(cluster_labels)
        train_clusters, test_clusters = train_test_split(
            unique_clusters, test_size=config.TEST_SIZE, random_state=42
        )
        train_idx = np.isin(cluster_labels, train_clusters)
        test_idx = np.isin(cluster_labels, test_clusters)

        # Now use full X (with ID_COLUMN) for model input/output
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        raise ValueError(
            "Split strategy !" + str(config.SPLIT_STRATEGY) + "! is invalid"
        )

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


def get_split_iterator(X, stratify_col):
    if config.SPLIT_STRATEGY.name == SplitStrategy.STRATIFY.name:
        return StratifiedKFold(
            n_splits=config.NUMBER_OF_FOLDS, shuffle=True, random_state=42
        ).split(X, stratify_col)
    elif config.SPLIT_STRATEGY.name == SplitStrategy.RANDOM.name:
        return KFold(
            n_splits=config.NUMBER_OF_FOLDS, shuffle=True, random_state=42
        ).split(X)
    elif config.SPLIT_STRATEGY.name == SplitStrategy.CLUSTER.name:
        cluster_labels = KMeans(
            n_clusters=int(len(X) / 10), random_state=42
        ).fit_predict(X)
        unique_clusters = np.unique(cluster_labels)
        kf = KFold(n_splits=config.NUMBER_OF_FOLDS, shuffle=True, random_state=42)
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


def average_per_organism_results(per_fold_results_list):
    merged = {}

    for fold_result in per_fold_results_list:
        for org_code, metrics in fold_result.items():
            if org_code not in merged:
                merged[org_code] = defaultdict(list)
            for k, v in metrics.items():
                merged[org_code][k].append(v)

    averaged = {}
    for org_code, metrics in merged.items():
        averaged[org_code] = {k: np.nanmean(v_list) for k, v_list in metrics.items()}

    return averaged


def run_cross_validated_random_forest(preprocessed_data, rf_settings_cv):
    cv_results = []
    test_label_counts = []
    train_label_counts = []
    org_res = {}

    for col in preprocessed_data.target_cols:

        # Remove rows with label 0 (unlabeled)
        df = preprocessed_data.merged_input[preprocessed_data.merged_input[col] != 0]
        # Remove rare Organism_Code values (only occur once)
        organism_counts = df[ORGANISM_COLUMN].value_counts()
        common_organisms = organism_counts[organism_counts > 1].index
        df = df[df[ORGANISM_COLUMN].isin(common_organisms)]

        X = df[preprocessed_data.feature_cols]
        y = df[col]
        stratify_col = df[ORGANISM_COLUMN]

        split_iterator = get_split_iterator(X, stratify_col)

        fold_per_organism_results = defaultdict(list)

        idx = 0
        for train_idx, test_idx in split_iterator:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if set(y_train.unique()) != set(y_test.unique()):
                print(
                    f"Skipped fold for {col}: y_train and y_test have different classes."
                )
                continue

            result = run_random_forest(
                X_train, y_train, X_test, y_test, col, rf_settings_cv
            )

            # Add results to list
            if len(cv_results) <= idx:
                cv_results.append({})
                test_label_counts.append({})
                train_label_counts.append({})
            (cv_results[idx])[col] = result
            (test_label_counts[idx])[col] = y_test.value_counts()
            (train_label_counts[idx])[col] = y_train.value_counts()
            idx = idx + 1

            organism_test = df.iloc[test_idx][ORGANISM_COLUMN]
            per_fold_result = evaluate_per_organism(
                result.y_pred, y_test, organism_test, result.y_score, col
            )
            fold_per_organism_results[col].append(per_fold_result)

        org_res[col] = average_per_organism_results(fold_per_organism_results[col])

    return cv_results, test_label_counts, train_label_counts, org_res


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


def evaluation_to_csv(results_dto_list, y_test_input_list, y_train_input_list):

    evaluation_df_list = []
    for idx, results_dto in enumerate(results_dto_list):
        y_test_input = y_test_input_list[idx]
        y_train_input = y_train_input_list[idx]
        evaluation_df = pd.DataFrame(
            columns=[
                "Accuracy",
                "ROC_Mean",
                "PR_Mean",
                "Precision_Mean",
                "Recall_Mean",
                "F1_Mean",
                "VME",
                "ME",
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
                result.vme,
                result.me,
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
            median(evaluation_df["VME"].dropna()),
            median(evaluation_df["ME"].dropna()),
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ]
        evaluation_df_list.append(evaluation_df)

    evaluation_df_3d = np.array([df.values for df in evaluation_df_list])

    mean_array = np.nanmean(evaluation_df_3d, axis=0)
    std_array = np.nanstd(evaluation_df_3d, axis=0)

    mean_df = pd.DataFrame(
        mean_array,
        index=evaluation_df_list[0].index,
        columns=evaluation_df_list[0].columns,
    )
    std_df = pd.DataFrame(
        std_array,
        index=evaluation_df_list[0].index,
        columns=evaluation_df_list[0].columns,
    )

    print("Standard Deviation:")
    print(std_df)

    print("Mean Values:")
    print(mean_df)

    os.makedirs("Evaluation", exist_ok=True)

    mean_df.to_csv(
        "Evaluation/Evaluation.csv", index=True, header=True, index_label="Target"
    )
    std_df.to_csv(
        "Evaluation/Standard_Deviation.csv",
        index=True,
        header=True,
        index_label="Target",
    )


def per_organism_evaluation_to_csv(
    results_organism, output_path="Evaluation/Per_Organism_Evaluation.csv"
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_metrics = set()
    rows = []
    for target, org_dict in results_organism.items():
        for organism, metrics in org_dict.items():
            row = {"Target": target, "Organism": organism}
            for metric_name, value in metrics.items():
                row[metric_name] = value
                all_metrics.add(metric_name)
            rows.append(row)

    df = pd.DataFrame(rows)

    for metric in all_metrics:
        metric_df = df.pivot(index="Target", columns="Organism", values=metric)
        if not metric_df.empty:
            metric_df.loc["Median"] = metric_df.median(numeric_only=True)
            metric_output_path = output_path.replace(".csv", f"_{metric}.csv")
            metric_df.to_csv(metric_output_path)
            print(f"{metric.title()}-table saved at: {metric_output_path}")


def tune_hyperparameter(preprocessed_data):
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

        rf_cv_results, _, _, _ = run_cross_validated_random_forest(
            preprocessed_data,
            rf_settings,
        )

        accs, rocs, prs, precs, recs, f1s = [], [], [], [], [], []
        for result_dic in rf_cv_results:
            for result in result_dic.values():
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


def feature_importance_to_csv(results_dto_list):
    importance_df_list = []
    for results_dto in results_dto_list:
        importance_df = pd.DataFrame(columns=results_dto.keys())
        importance_df[ORGANISM_COLUMN] = None
        for name in results_dto:
            importance_df.loc[name] = 0.0

            def strip_prefix(name: str) -> str:
                prefixes = ("S_", "I_", "R_")
                for p in prefixes:
                    if name.startswith(p):
                        return name[len(p) :]
                return name

            result = results_dto[name]
            s_clean = result.feature_importance.rename(index=strip_prefix)
            s_clean = s_clean.groupby(s_clean.index).sum()
            importance_df.loc[name, s_clean.index] += s_clean
        importance_df_list.append(importance_df)
    pd.concat(importance_df_list).groupby(level=0).mean().to_csv(
        "Evaluation/feature_importance.csv"
    )


def generate_prediction_results(
    dataset_input, preprocessed_input, rf_results_input, real_path
):
    pred_dict = {name: result.y_pred for name, result in rf_results_input[0].items()}
    pred_df = pd.DataFrame(pred_dict)

    inv_mapping = {v: k for k, v in RESISTANCE_MAPPING.items()}
    pred_df = pred_df.replace(inv_mapping)

    pred_df[ID_COLUMN] = preprocessed_input.merged_input[ID_COLUMN]
    cols = [ID_COLUMN] + [col for col in pred_df.columns if col != ID_COLUMN]
    pred_df = pred_df[cols]
    path = Path(dataset_input["PathToCsv"][0])
    new_path = path.parent / f"Result_{path.name}"
    pred_df.to_csv(new_path, index=False)
    print(pred_df)

    if config.EVALUATE_PREDICTION:
        accuracy_df = calc_accuracy(real_path, pred_df)
        print(accuracy_df)
        evaluation_path = path.parent / f"Evaluation_{path.name}"
        accuracy_df.to_csv(evaluation_path, index=False)


def calc_accuracy(real_path, pred_df):
    real_df = pd.read_csv(
        real_path.iloc[0],
        na_values=["NA", "NaN", "nan", "N/A", ""],
        keep_default_na=True,
    )

    # Clean up
    real_df = real_df.drop(columns=["Sample_ID_IfH", "Organism_Code"])
    pred_df = pred_df.drop(columns=["Sample_ID_IfH"])
    pred_df.columns = [c.replace("_AB", "") for c in pred_df.columns]

    # Find common antibiotics
    common_abs = sorted(set(real_df.columns) & set(pred_df.columns))
    print("Common antibiotics:", common_abs)
    real_df = real_df[common_abs]
    pred_df = pred_df[common_abs]

    # Compute per-antibiotic accuracy
    results = []
    for ab in common_abs:
        mask = real_df[ab].notna()  # ignore NA values in real
        if mask.sum() > 0:
            acc = (real_df.loc[mask, ab] == pred_df.loc[mask, ab]).mean()
            results.append({"antibiotic": ab, "n_samples": mask.sum(), "accuracy": acc})
        else:
            results.append({"antibiotic": ab, "n_samples": 0, "accuracy": None})
    accuracy_df = pd.DataFrame(results)

    # Remove rows with missing accuracy
    valid_df = accuracy_df.dropna(subset=["accuracy"])

    # Calc weighted median
    sorted_df = valid_df.sort_values("accuracy")
    acc_values = sorted_df["accuracy"].to_numpy()
    weights = sorted_df["n_samples"].to_numpy()

    def weighted_median(values, weights):
        sorter = np.argsort(values)
        values, weights = values[sorter], weights[sorter]
        cumsum = np.cumsum(weights)
        cutoff = weights.sum() / 2
        return values[np.searchsorted(cumsum, cutoff)]

    wm = weighted_median(acc_values, weights)

    # Add weighted median
    accuracy_df = pd.concat(
        [
            accuracy_df,
            pd.DataFrame(
                [
                    {
                        "antibiotic": "Weighted_median",
                        "n_samples": weights.sum(),
                        "accuracy": wm,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return accuracy_df


def process(dataset_list_input):
    rf_settings_input = RandomForestSettings(
        n_estimators=100,
        class_weight="balanced_subsample",
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.3,
        bootstrap=True,
    )

    rf_settings_stacked = RandomForestSettings(
        n_estimators=1500,
        class_weight="balanced_subsample",
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.3,
        bootstrap=True,
    )

    data_loader = preprocessing.DataLoader()

    if config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
        preprocessed_data_input = data_loader.get_genotype_data_for_prediction(
            dataset_list_input
        )

    else:
        preprocessed_data_input = data_loader.get_preprocessed_data(dataset_list_input)
        preprocessed_data_input = filter_merged_input(preprocessed_data_input, 20)

    start_time = time.time()
    if config.EXECUTION_MODE == ExecutionMode.TUNE_HYPERPARAMETER:
        tune_hyperparameter(preprocessed_data_input)

    else:
        if not config.STACK_MODEL and not config.CROSS_VALIDATE:
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
                per_organism_results,
            ) = run_splitted_random_forest(
                preprocessed_data_input,
                rf_settings_input,
            )

            display_results(rf_results[0], False)

        elif not config.STACK_MODEL and config.CROSS_VALIDATE:
            # display not possible with CV.
            # S/R/I Set changes with every fold -> fpr size changes as well
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
                per_organism_results,
            ) = run_cross_validated_random_forest(
                preprocessed_data_input,
                rf_settings_input,
            )

        elif config.STACK_MODEL and not config.CROSS_VALIDATE:
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
                per_organism_results,
            ) = run_stacked_random_forest(
                preprocessed_data_input,
                rf_settings_input,
                rf_settings_stacked,
                False,
            )

            if config.EXECUTION_MODE == ExecutionMode.PREDICT_ON_SAVED:
                generate_prediction_results(
                    dataset_list_input,
                    preprocessed_data_input,
                    rf_results,
                    dataset_list_input["PathToCsv"],
                )

            else:
                display_results(rf_results[0], False)
                feature_importance_to_csv(rf_results)

        elif config.STACK_MODEL and config.CROSS_VALIDATE:
            (
                rf_results,
                y_test_count_result,
                y_train_count_result,
                per_organism_results,
            ) = run_stacked_random_forest(
                preprocessed_data_input,
                rf_settings_input,
                rf_settings_stacked,
                True,
            )

        else:
            raise ValueError("Model strategy is invalid")
        if config.EXECUTION_MODE != ExecutionMode.PREDICT_ON_SAVED:
            per_organism_evaluation_to_csv(per_organism_results)
            evaluation_to_csv(rf_results, y_test_count_result, y_train_count_result)
            if config.STACK_MODEL:
                feature_importance_to_csv(rf_results)
        print("--- %s seconds for ML---" % (time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RF on multiple datasets from a settings file"
    )
    parser.add_argument("path_file", help="Path to the settings CSV file")

    args = parser.parse_args()

    dataset_list = load_dataset_paths(args.path_file)

    process(dataset_list)
