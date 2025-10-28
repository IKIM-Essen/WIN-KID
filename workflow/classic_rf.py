import os

import pandas as pd
from constants import ORGANISM_COLUMN
from constants import MODEL_FOLDER

import config
import utils
from split_strategies import SplitStrategy
import numpy as np
from execution_modes import ExecutionMode
import cloudpickle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def filter_preprocessed_data(merged_filtered_input, col):
    # Remove rows with label 0 (unlabeled)
    merged_filtered_input = merged_filtered_input[merged_filtered_input[col] != 0]

    # Drop rows of Organisms that occur only once
    value_counts = merged_filtered_input[ORGANISM_COLUMN].value_counts()
    rare_values = value_counts[value_counts == 1].index
    merged_filtered_input = merged_filtered_input[
        ~merged_filtered_input[ORGANISM_COLUMN].isin(rare_values)
    ]
    return merged_filtered_input


def split_train_test(merged_filtered_input, X, y):
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
    else:
        raise ValueError("SPLIT_STRATEGY is invalid ")

    return y_test, y_train, X_test, X_train


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
        utils.validate_target_values(y_train_input, y_test_input, target_input)
        model.fit(X_train_input, y_train_input)

    elif config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
        utils.validate_target_values(y_train_input, y_test_input, target_input)
        model.fit(X_train_input, y_train_input)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)

    elif (
        config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE
        or config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE
    ):
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

    return utils.generate_result(
        target_input,
        y_test_input,
        proba_full,
        y_pred,
        forest_importances.sort_values(ascending=False),
    )


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
