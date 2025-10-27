from collections import Counter
import logging
import os
from statistics import mean
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from constants import ID_COLUMN, MODEL_FOLDER, ORGANISM_COLUMN, RESISTANCE_MAPPING
import utils
import config
from execution_modes import ExecutionMode
from split_strategies import SplitStrategy

logger = logging.getLogger(__name__)


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
        utils.validate_target_values(y_train_input, y_val_input, target_input)
        model.fit(X_train_input, y_train_input)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)

    elif (
        config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE
        or config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE
    ):
        X_test_input = X_test_input.drop(ID_COLUMN, axis=1)
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)

    elif config.EXECUTION_MODE == ExecutionMode.TRAIN_TEST:
        utils.validate_target_values(y_train_input, y_val_input, target_input)
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


def get_stratified_split(preprocessed_data):
    skf = StratifiedKFold(
        n_splits=config.NUMBER_OF_FOLDS, shuffle=True, random_state=42
    )
    stratify_col = preprocessed_data.merged_input[ORGANISM_COLUMN]
    return skf, stratify_col


def split_stacked_rf(X, y):
    if (
        config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE
        or config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE
    ):
        logger.warning(
            "In %s all samples are used for test", config.EXECUTION_MODE.value
        )
        y_test = y
        y_train = y
        X_test = X
        X_train = X
    else:
        y_test, y_train, X_test, X_train = split_sets_for_stacked(X, y)
        if config.EXECUTION_MODE == ExecutionMode.TRAIN_TEST:
            os.makedirs("Evaluation", exist_ok=True)
            X_train[ID_COLUMN].to_csv(
                "Evaluation/sorted_samples_train.csv",
                index=False,
            )
            X_test[ID_COLUMN].to_csv(
                "resources/sorted_samples_test.csv",
                index=False,
            )

        # Train with all sample if saved
    if config.EXECUTION_MODE == ExecutionMode.SAVE_TRAINED:
        logger.warning(
            "n %s all samples are used for training", config.EXECUTION_MODE.value
        )
        y_train = pd.concat([y_train, y_test], ignore_index=True)
        X_train = pd.concat([X_train, X_test], ignore_index=True)
    return X_train, X_test, y_train, y_test


def prepare_first_layer(preprocessed_data):
    merged_filtered_input = preprocessed_data.merged_input

    if (
        config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_SAVE
        or config.EXECUTION_MODE != ExecutionMode.PREDICT_AND_EVALUATE
    ):
        # Drop rows of Organisms that occur only once
        value_counts = preprocessed_data.merged_input[ORGANISM_COLUMN].value_counts()
        rare_values = value_counts[value_counts == 1].index
        merged_filtered_input = merged_filtered_input[
            ~merged_filtered_input[ORGANISM_COLUMN].isin(rare_values)
        ]

    feature_cols_with_id = preprocessed_data.feature_cols + [ID_COLUMN]
    X = merged_filtered_input[feature_cols_with_id]
    y = merged_filtered_input[preprocessed_data.target_cols]
    return merged_filtered_input, X, y


def filter_na_values(
    y_proba_train, X_proba_train, y_proba_test, X_proba_test, organism_test, target
):
    train_mask = y_proba_train[target] != 0
    X_train_filtered = X_proba_train[train_mask]
    y_train_filtered = y_proba_train[target][train_mask]
    test_mask = y_proba_test[target] != 0
    X_test_filtered = X_proba_test[test_mask]
    y_test_filtered = y_proba_test[target][test_mask]
    organism_test_filtered = organism_test[test_mask]

    if config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
        X_train_filtered = X_proba_train
        y_train_filtered = y_proba_train[target]
        X_test_filtered = X_proba_test
        y_test_filtered = y_proba_test[target]

    if (
        config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE
        or config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE
    ):
        X_train_filtered = X_test_filtered
        y_train_filtered = y_test_filtered
    return (
        X_train_filtered,
        y_train_filtered,
        X_test_filtered,
        y_test_filtered,
        organism_test_filtered,
    )


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
        if (
            config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE
            or config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE
        ):
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
        config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE
        or config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_EVALUATE
    ) and len(X_train_target) < 5:
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


def compute_label_distribution(label_counts_list):
    dict_list = [
        s.to_dict() if isinstance(s, pd.Series) else s for s in label_counts_list
    ]
    transposed = {key: [d[key] for d in dict_list] for key in dict_list[0]}
    return {key: mean(values) for key, values in transposed.items()}
