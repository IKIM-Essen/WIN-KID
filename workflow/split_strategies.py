from enum import Enum


class SplitStrategy(Enum):
    RANDOM = "random"
    STRATIFY = "stratify"
    CLUSTER = "cluster"
