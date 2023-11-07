"""Global settings."""

import os
from pathlib import Path
from typing import Optional


MINIMUM_PAGE_LENGTH: int = 5
"""Shorter texts are considered as empty."""

EMPTY_PAGE_OUTPUT: Optional[int] = 0
"""Output value for empty pages.
If None, empty pages are handled through the standard pipeline."""

SHORT_COLUMN_WIDTH: int = 5
"""If all lines (columns) in a page are shorter than this it is considered broken."""

ENCODING = "utf-8"
"""Encoding to be used throughout all text file processing operations."""

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LINE_SEPARATOR = os.getenv("LINE_SEPARATOR", "\n")

Q_GRAM_LENGTH: int = int(os.environ.get("Q_GRAM_LENGTH", "3"))
Q_GRAMS_GAMMA: int = int(os.environ.get("Q_GRAMS_GAMMA", "1000"))

SOURCE_DIR = Path(__file__).parent
DATA_DIR = SOURCE_DIR / "data"

DICTS_DIR = DATA_DIR / "dicts"
HUNSPELL_DIR = DICTS_DIR / "hunspell"

QGRAMS_DIR = DATA_DIR / "qgrams"

CLASSIFIER_DIR = DATA_DIR / "classifier"

for directory in (DATA_DIR, DICTS_DIR, HUNSPELL_DIR, CLASSIFIER_DIR):
    if not directory.is_dir():
        raise NotADirectoryError(directory)


### INITIALIZE
DEFAULT_LANGUAGE = "nl"

HUNSPELL_LANGUAGE = DEFAULT_LANGUAGE
TOKEN_DICT_FILE: Path = DICTS_DIR / "nl_voc.txt"
QGRAMS_FILE: Path = QGRAMS_DIR / "nl_voc.txt"
PIPELINE_FILE: Path = CLASSIFIER_DIR / "pipeline_nn.joblib"

for file in (TOKEN_DICT_FILE, QGRAMS_FILE, PIPELINE_FILE):
    if not file.is_file():
        raise FileNotFoundError(file)
