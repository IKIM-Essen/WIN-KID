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
