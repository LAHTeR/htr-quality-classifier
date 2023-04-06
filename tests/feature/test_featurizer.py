import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from text_quality.feature.featurize import Scorers


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
