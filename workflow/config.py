# workflow/config.py
from modes import Mode
from split_strategies import SplitStrategy


STACK_MODEL = True
CROSS_VALIDATE = False
NUMBER_OF_FOLDS = 5
TEST_SIZE = 0.3
SPLIT_STRATEGY = SplitStrategy.STRATIFY
MODE = Mode.PREDICT_ON_SAVED
