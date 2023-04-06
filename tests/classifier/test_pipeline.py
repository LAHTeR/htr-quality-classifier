import joblib
import pytest
import sklearn
from text_quality.classifier.pipeline import ClassifierScores
from text_quality.classifier.pipeline import Pipeline
from text_quality.classifier.settings import PIPELINE_FILE
from text_quality.feature.featurize import Scorers


@pytest.fixture
def sklearn_pipeline() -> sklearn.pipeline.Pipeline:
    return joblib.load(PIPELINE_FILE)


@pytest.fixture
def pipeline(featurizer) -> Pipeline:
    return Pipeline.from_file(PIPELINE_FILE, featurizer)


class TestPipeline:
    def test_from_file(self, sklearn_pipeline, featurizer):
        # pylint: disable=protected-access
        assert (
            Pipeline.from_file(PIPELINE_FILE, featurizer)._pipeline.feature_names_in_
            == sklearn_pipeline.feature_names_in_
        ).all()

    def test_features(self, pipeline):
        assert pipeline.features == list(Scorers.__annotations__.keys())

    @pytest.mark.parametrize("text, expected", [("", 3), ("een Nederlands tekst", 1)])
    def test_classify(self, pipeline, text, expected):
        assert pipeline.classify(text) == expected

    @pytest.mark.parametrize(
        "text, expected_class, expected_scores",
        [
            (
                "",
                3,
                ClassifierScores(
                    confidence=0.9979264598992952,
                    dict_score=0,
                    dict_score_gt=0,
                    n_gram_score=0,
                    garbage_score=0,
                    n_characters=0,
                    n_tokens=0,
                ),
            ),
            (
                "een Nederlands tekst",
                1,
                ClassifierScores(
                    confidence=0.728276021849431,
                    dict_score=1,
                    dict_score_gt=0.7222222222222222,
                    n_gram_score=0.347,
                    garbage_score=0,
                    n_characters=20,
                    n_tokens=3,
                ),
            ),
        ],
    )
    def test_classify_with_scores(
        self, pipeline, text, expected_class, expected_scores
    ):
        quality, scores = pipeline.classify_with_scores(text)
        assert quality == expected_class
        assert scores == pytest.approx(expected_scores)
