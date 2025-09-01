# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

from enum import Enum


class SplitStrategy(Enum):
    RANDOM = "random"
    STRATIFY = "stratify"
    CLUSTER = "cluster"
