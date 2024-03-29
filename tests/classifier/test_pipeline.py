from contextlib import nullcontext as does_not_raise
import joblib
import pytest
import sklearn
from pagexml.model.physical_document_model import PageXMLScan
from pagexml.model.physical_document_model import PageXMLTextLine
from text_quality.classifier.pipeline import ClassifierScores, Reason
from text_quality.classifier.pipeline import Pipeline
from text_quality.classifier.pipeline import default_scores_dict
from text_quality.feature.featurize import Scorers
from text_quality.page.page import Page
from text_quality.settings import PIPELINE_FILE


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

    @pytest.mark.parametrize(
        "page,expected",
        [
            ("", 0),
            ("een Nederlandse tekst", 1),
            (Page(PageXMLScan()), 0),
            (
                Page(PageXMLScan(lines=[PageXMLTextLine(text="test")])),
                3,
            ),
            (
                Page(PageXMLScan(lines=[PageXMLTextLine(text="een Nederlands tekst")])),
                1,
            ),
            (
                Page(PageXMLScan(lines=[PageXMLTextLine(text="test")] * 10)),
                3,
            ),
        ],
    )
    def test_classify(self, pipeline, page, expected):
        assert pipeline.classify(page) == expected

    @pytest.mark.parametrize(
        "text, expected_class, expected_scores, expected_reason",
        [
            (
                "",
                0,
                ClassifierScores(
                    confidence=1.0,
                    dict_score=0,
                    dict_score_gt=0,
                    n_gram_score=0,
                    garbage_score=0,
                    n_characters=0,
                    n_tokens=0,
                    language="0",
                    language_confidence=0.0,
                ),
                Reason.EMPTY,
            ),
            (
                "een Nederlandse tekst",
                1,
                ClassifierScores(
                    confidence=0.7212720237280937,
                    dict_score=1,
                    dict_score_gt=0.7368421052631579,
                    n_gram_score=0.3203076923076923,
                    garbage_score=0,
                    n_characters=21,
                    n_tokens=3,
                    language="nl",
                    language_confidence=1.0,
                ),
                Reason.CLASSIFIER,
            ),
            (
                Page(PageXMLScan(lines=[PageXMLTextLine(text="test")])),
                3,
                ClassifierScores(
                    confidence=1,
                    dict_score=0,
                    dict_score_gt=0,
                    n_gram_score=0,
                    garbage_score=0,
                    n_characters=4,
                    n_tokens=0,
                    language="0",
                    language_confidence=0.0,
                ),
                Reason.SHORT_COLUMNS,
            ),
            (
                Page(
                    PageXMLScan(lines=[PageXMLTextLine(text="een Nederlandse tekst")])
                ),
                1,
                ClassifierScores(
                    confidence=0.7212720237280937,
                    dict_score=1,
                    dict_score_gt=0.7368421052631579,
                    n_gram_score=0.3203076923076923,
                    garbage_score=0,
                    n_characters=21,
                    n_tokens=3,
                    language="nl",
                    language_confidence=1.0,
                ),
                Reason.CLASSIFIER,
            ),
            (
                Page(PageXMLScan(lines=[PageXMLTextLine(text="test")] * 10)),
                3,
                ClassifierScores(
                    confidence=1,
                    dict_score=0,
                    dict_score_gt=0,
                    n_gram_score=0,
                    garbage_score=0,
                    n_characters=49,
                    n_tokens=0,
                    language="0",
                    language_confidence=0.0,
                ),
                Reason.SHORT_COLUMNS,
            ),
        ],
    )
    # pylint: disable=too-many-arguments
    def test_classify_with_scores(
        self, pipeline, text, expected_class, expected_scores, expected_reason
    ):
        quality, scores, reason = pipeline.classify_with_scores(text)
        assert quality == expected_class
        assert scores == pytest.approx(expected_scores)
        assert reason == expected_reason


@pytest.mark.parametrize(
    "default_value, fields, expected, expected_exception",
    [
        (
            0,
            {},
            ClassifierScores(
                confidence=0.0,
                n_characters=0,
                n_tokens=0,
                dict_score=0.0,
                dict_score_gt=0.0,
                n_gram_score=0.0,
                garbage_score=0.0,
                language="0",
                language_confidence=0.0,
            ),
            does_not_raise(),
        ),
        (
            1,
            {},
            ClassifierScores(
                confidence=1.0,
                n_characters=1,
                n_tokens=1,
                dict_score=1.0,
                dict_score_gt=1.0,
                n_gram_score=1.0,
                garbage_score=1.0,
                language="1",
                language_confidence=1.0,
            ),
            does_not_raise(),
        ),
        (
            0,
            {"confidence": 1.0, "n_characters": 1},
            ClassifierScores(
                confidence=1.0,
                n_characters=1,
                n_tokens=0,
                dict_score=0.0,
                dict_score_gt=0.0,
                n_gram_score=0.0,
                garbage_score=0.0,
                language="0",
                language_confidence=0.0,
            ),
            does_not_raise(),
        ),
        (
            0,
            {"confidence": 1.0, "n_characters": 1, "invalid": 0},
            None,
            pytest.raises(ValueError, match=r"Unknown field 'invalid'.*"),
        ),
    ],
)
def test_default_scores_dict(default_value, fields, expected, expected_exception):
    with expected_exception:
        assert default_scores_dict(default_value, **fields) == expected
