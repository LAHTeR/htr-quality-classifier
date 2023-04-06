from typing import List
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

    @property
    def features(self) -> List[str]:
        return list(self._scorers.keys())

    def featurize(self, text: str) -> tuple[dict[str, float], List[str]]:
        tokens = self._tokenizer.tokenize(text)
        return {
            feature: scorer.score(tokens) for feature, scorer in self._scorers.items()
        }, tokens

    def featurize_as_dataframe(self, text: str) -> tuple[pd.DataFrame, List[str]]:
        features, tokens = self.featurize(text)
        return Featurizer.as_dataframe(features), tokens

    @staticmethod
    def as_dataframe(features: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame({feature: [value] for feature, value in features.items()})
