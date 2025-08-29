# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from execution_modes import ExecutionMode
from split_strategies import SplitStrategy


STACK_MODEL = True
CROSS_VALIDATE = False
NUMBER_OF_FOLDS = 5
TEST_SIZE = 0.3
SPLIT_STRATEGY = SplitStrategy.STRATIFY
EXECUTION_MODE = ExecutionMode.PREDICT_ON_SAVED
