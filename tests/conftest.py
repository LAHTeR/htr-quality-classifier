import pytest
from text_quality.classifier.settings import HUNSPELL_DIR
from text_quality.classifier.settings import HUNSPELL_LANGUAGE
from text_quality.classifier.settings import QGRAMS_FILE
from text_quality.classifier.settings import TOKEN_DICT_FILE
from text_quality.feature.featurize import Featurizer
from text_quality.feature.featurize import Scorers
from text_quality.feature.scorer.dictionary import HunspellDictionary
from text_quality.feature.scorer.dictionary import TokenDictionary
from text_quality.feature.scorer.garbage import GarbageDetector
from text_quality.feature.scorer.q_gram import QGram
from text_quality.feature.tokenizer import NautilusOcrTokenizer


@pytest.fixture
def featurizer():
    scorers = Scorers(
        dict_score=HunspellDictionary.from_path(HUNSPELL_DIR, HUNSPELL_LANGUAGE),
        dict_score_gt=TokenDictionary.from_file(TOKEN_DICT_FILE),
        n_gram_score=QGram.from_file(QGRAMS_FILE),
        garbage_score=GarbageDetector(),
    )
    tokenizer = NautilusOcrTokenizer()
    return Featurizer(scorers, tokenizer)
