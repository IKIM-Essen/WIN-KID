# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from enum import Enum


class ExecutionMode(Enum):
    SAVE_TRAINED = "SAVE_TRAINED"
    PREDICT_ON_SAVED = "PREDICT_ON_SAVED"
    TRAIN_TEST = "TRAIN_TEST"
    TUNE_HYPERPARAMETER = "tune_hyperparameter"
