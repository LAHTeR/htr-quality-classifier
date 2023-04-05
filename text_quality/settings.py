import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LINE_SEPARATOR = os.getenv("LINE_SEPARATOR", "\n")

Q_GRAM_LENGTH: int = int(os.environ.get("Q_GRAM_LENGTH", "3"))
Q_GRAMS_GAMMA: int = int(os.environ.get("Q_GRAMS_GAMMA", "1000"))

CWD = Path(__file__).parent.absolute()

DATA_DIR = CWD / "data"

DICTS_DIR = DATA_DIR / "dicts"
HUNSPELL_DIR = DICTS_DIR / "hunspell"

QGRAMS_DIR = DATA_DIR / "qgrams"

CLASSIFIER_DIR = DATA_DIR / "classifier"

for dir in (DATA_DIR, DICTS_DIR, HUNSPELL_DIR, CLASSIFIER_DIR):
    if not dir.is_dir():
        raise NotADirectoryError(dir)


### INITIALIZE
HUNSPELL_LANGUAGE = "nl"
TOKEN_DICT_FILE: Path = DICTS_DIR / "nl_voc.txt"
QGRAMS_FILE: Path = QGRAMS_DIR / "nl_voc.txt"
PIPELINE_FILE: Path = CLASSIFIER_DIR / "pipeline_nn.joblib"

for file in (TOKEN_DICT_FILE, QGRAMS_FILE, PIPELINE_FILE):
    if not file.is_file():
        raise FileNotFoundError(file)
