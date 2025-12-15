# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from enum import Enum


class ExecutionMode(Enum):
    SAVE_TRAINED = "SAVE_TRAINED"
    PREDICT_AND_SAVE = "PREDICT_AND_SAVE"
    PREDICT_AND_EVALUATE = "PREDICT_AND_EVALUATE"
    TRAIN_TEST = "TRAIN_TEST"
    TUNE_HYPERPARAMETER = "tune_hyperparameter"

class W2VMode(Enum):
    TRAIN_W2V = "TRAIN_W2V"
    TUNE_W2V = "TUNE_W2V"