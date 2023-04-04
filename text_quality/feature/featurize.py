from typing import TypedDict
import pandas as pd
from .scorer.dictionary import HunspellDictionary
from .scorer.dictionary import TokenDictionary
from .scorer.garbage import GarbageDetector
from .scorer.q_gram import QGram
from .tokenizer import Tokenizer


class Scorers(TypedDict):
    """A configuration of features and respective Scorers"""

    dict_score: HunspellDictionary
    dict_score_gt: TokenDictionary
    n_gram_score: QGram
    garbage_score: GarbageDetector


class Featurizer:
    """A collection of scorers to featurize an input text."""

    def __init__(self, scorers: Scorers, tokenizer: Tokenizer) -> None:
        self._scorers = scorers
        self._tokenizer = tokenizer

    def featurize(self, text: str) -> dict[str, float]:
        tokens = self._tokenizer.tokenize(text)
        return {
            feature: scorer.score(tokens) for feature, scorer in self._scorers.items()
        }

    def featurize_as_dataframe(self, text: str) -> pd.DataFrame:
        return pd.DataFrame(
            {feature: [value] for feature, value in self.featurize(text).items()}
        )
