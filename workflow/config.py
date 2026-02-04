# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import logging
from execution_modes import ExecutionMode
from execution_modes import W2VMode
from split_strategies import SplitStrategy


STACK_MODEL = True
CROSS_VALIDATE = False
NUMBER_OF_FOLDS = 5
TEST_SIZE = 0.2
SPLIT_STRATEGY = SplitStrategy.STRATIFY
EXECUTION_MODE = ExecutionMode.PREDICT_AND_SAVE
MIN_SAMPLE_NUMBER = 15
LOGGING_LEVEL = logging.INFO
KMERE = False
W2V_MODE = W2VMode.TUNE_W2V
FASTA_DIR = ""
RETRAIN_W2V = False  # Set to True to retrain model
