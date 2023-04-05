import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from text_quality.feature.featurize import Featurizer
from text_quality.feature.featurize import Scorers
from text_quality.feature.scorer.dictionary import HunspellDictionary
from text_quality.feature.scorer.dictionary import TokenDictionary
from text_quality.feature.scorer.garbage import GarbageDetector
from text_quality.feature.scorer.q_gram import QGram
from text_quality.feature.tokenizer import NautilusOcrTokenizer
from text_quality.settings import HUNSPELL_DIR
from text_quality.settings import HUNSPELL_LANGUAGE
from text_quality.settings import QGRAMS_FILE
from text_quality.settings import TOKEN_DICT_FILE


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


class TestFeaturizer:
    def test_features(self, featurizer):
        assert featurizer.features == list(Scorers.__annotations__.keys())

    @pytest.mark.parametrize(
        "text, expected_features, expected_tokens",
        [
            (
                "",
                {feature: 0 for feature in Scorers.__annotations__.keys()},
                [],
            ),
            (
                "test token",
                {
                    "dict_score": 1,
                    "dict_score_gt": 0,
                    "n_gram_score": 0.3402,
                    "garbage_score": 0,
                },
                ["test", "token"],
            ),
        ],
    )
    def test_featurize(self, featurizer, text, expected_features, expected_tokens):
        features, tokens = featurizer.featurize(text)
        assert features == expected_features
        assert tokens == expected_tokens

    @pytest.mark.parametrize(
        "text, expected_df, expected_tokens",
        [
            (
                "",
                pd.DataFrame([[0, 0, 0, 0]], columns=Scorers.__annotations__.keys()),
                [],
            ),
            (
                "test token",
                pd.DataFrame(
                    [[1, 0, 0.3402, 0]], columns=Scorers.__annotations__.keys()
                ),
                ["test", "token"],
            ),
        ],
    )
    def test_featurize_as_dataframe(
        self, featurizer, text, expected_df, expected_tokens
    ):
        features_df, tokens = featurizer.featurize_as_dataframe(text)
        assert_frame_equal(features_df, expected_df, check_dtype=False)
        assert tokens == expected_tokens
