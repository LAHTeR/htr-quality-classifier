import os
from pathlib import Path

LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO")

Q_GRAM_LENGTH: int = int(os.environ.get("Q_GRAM_LENGTH", "3"))
Q_GRAMS_GAMMA: int = int(os.environ.get("Q_GRAMS_GAMMA", "1000"))

CWD = Path(__file__).parent.absolute()

DATA_DIR = CWD / "data"

DICTS_DIR = DATA_DIR / "dicts"
HUNSPELL_DIR = DICTS_DIR / "hunspell"

QGRAMS_DIR = DATA_DIR / "qgrams"

CLASSIFIER_DIR = DATA_DIR / "classifier"

for dir in (DATA_DIR, DICTS_DIR, HUNSPELL_DIR, CLASSIFIER_DIR):
    assert dir.is_dir(), f"Directory not found: '{dir.absolute()}'."
