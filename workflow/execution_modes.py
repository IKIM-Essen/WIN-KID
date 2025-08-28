from enum import Enum


class ExecutionMode(Enum):
    SAVE_TRAINED = "SAVE_TRAINED"
    PREDICT_ON_SAVED = "PREDICT_ON_SAVED"
    TRAIN_TEST = "TRAIN_TEST"
    TUNE_HYPERPARAMETER = "tune_hyperparameter"
