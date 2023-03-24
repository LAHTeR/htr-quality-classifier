from typing import TypedDict
from ..settings import CLASSIFIER_FEATURES
from .dictionary import HunspellDictionary
from .dictionary import TokenDictionary
from .garbage import GarbageDetector
from .q_gram import QGram


class FeatureTypes(TypedDict):
    dict_score_normalized: HunspellDictionary
    dict_score_gt_normalized: TokenDictionary
    n_gram_score_normalized: QGram
    garbage_score_normalized: GarbageDetector


class Featurizer:
    def __init__(self, featurizers: FeatureTypes) -> None:
        self._featurizers = featurizers
