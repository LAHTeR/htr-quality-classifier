import os
from pathlib import Path


Q_GRAM_LENGTH: int = os.environ.get("Q_GRAM_LENGTH")

CWD = Path(__file__).absolute
DATA_DIR = CWD / "data"
DICTS_DIR = DATA_DIR / "dicts"
HUNSPELL_DIR = DICTS_DIR / "hunspell"
CLASSIFIER_DIR = DATA_DIR / "classifier"

assert DATA_DIR.is_dir()
assert DICTS_DIR.is_dir()
assert HUNSPELL_DIR.is_dir()
assert CLASSIFIER_DIR.is_dir()

CLASSIFIER_FEATURES = [
    "dict_score_normalized",
    "dict_score_gt_normalized",
    "n_gram_score_normalized",
    "garbage_score_normalized",
]
