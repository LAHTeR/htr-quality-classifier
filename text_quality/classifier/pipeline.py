"""Classification pipeline."""

import logging
from enum import Enum
from enum import auto
from pathlib import Path
from typing import List
from typing import TypedDict
from typing import Union
import joblib
import pandas as pd
import sklearn.pipeline
from ..feature.featurize import Featurizer
from ..feature.featurize import Scorers
from ..language.fasttext import FastTextLanguageClassifier
from ..page.page import Page
from ..settings import DEFAULT_LANGUAGE
from ..settings import EMPTY_PAGE_OUTPUT
from ..settings import MINIMUM_PAGE_LENGTH
from ..settings import SHORT_COLUMN_WIDTH


ClassifierScores = TypedDict(
    "ClassifierScores",
    {
        "confidence": float,
        "n_characters": int,
        "n_tokens": int,
        "language": str,
        "language_confidence": float,
    }
    | {score: float for score in Scorers.__annotations__.keys()},
)
"""Container class for the scores returned by the classifier."""


class Reason(Enum):
    """Reasons for the classification result."""

    CLASSIFIER = auto()
    SHORT_COLUMNS = auto()
    EMPTY = auto()
    LANGUAGE = auto()  # language differs from default


def default_scores_dict(default_value, **fields) -> ClassifierScores:
    """Generate a ClassifierScores dict with default values.

    Args:
        default_value: The default value for the scores.
        fields: arguments to add to the dict, hence not taking the default value.
    """

    for field in fields:
        if field not in ClassifierScores.__annotations__:
            raise ValueError(
                f"Unknown field '{field}'. "
                f"Valid fields are: {ClassifierScores.__annotations__.keys()}"
            )

    return ClassifierScores(
        {
            field: _type(default_value)
            for field, _type in ClassifierScores.__annotations__.items()
        }
        | fields
    )


class Pipeline:
    """A wrapper around an sklearn pipeline that adds a featurizer."""

    def __init__(
        self,
        pipeline: sklearn.pipeline.Pipeline,
        featurizer: Featurizer,
        default_language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._pipeline = pipeline
        self._featurizer = featurizer
        self._default_language = default_language
        self._language_classifier = FastTextLanguageClassifier()

    @property
    def features(self) -> List[str]:
        """The names of the features used in the pipeline."""
        return list(self._pipeline.feature_names_in_)

    def classify(self, page: Union[Page, str]) -> int:
        """Single instance classification."""

        if isinstance(page, Page):
            quality = self._classify_pagexml(page)
        elif self._is_short(page):
            logging.debug(
                "Skipping short text: '%s' (%d characters).", page, len(page.strip())
            )
            quality = EMPTY_PAGE_OUTPUT
        else:
            language, _ = self._language_classifier.classify(page)
            if language == self._default_language:
                features, _ = self._featurizer.featurize_as_dataframe(page)
                quality = self._pipeline.predict(features)[0]
            else:
                logging.info(
                    "Language '%s' differs from default language '%s'.",
                    language,
                    self._default_language,
                )
                quality = EMPTY_PAGE_OUTPUT

        return quality

    def _classify_pagexml(self, pagexml: Page) -> int:
        """Classify a Page object."""

        if pagexml.lines() and all(
            len(line) < SHORT_COLUMN_WIDTH for line in pagexml.lines()
        ):
            logging.warning("Page '%s' has short columns.", pagexml.id)
            quality = 3
        else:
            quality = self.classify(pagexml.get_text())

        return quality

    def classify_with_scores(
        self, page: Union[Page, str]
    ) -> tuple[int, ClassifierScores, Reason]:
        """Single instance classification with scores."""

        if isinstance(page, Page):
            quality, scores, reason = self._classify_pagexml_with_scores(page)
        elif self._is_short(page):
            logging.debug(
                "Skipping short text: '%s' (%d characters).", page, len(page.strip())
            )
            quality = EMPTY_PAGE_OUTPUT
            scores = default_scores_dict(0, confidence=1.0, n_characters=len(page))
            reason = Reason.EMPTY
        else:
            language, language_confidence = self._language_classifier.classify(page)
            if language == self._default_language:
                features, tokens = self._featurizer.featurize(page)
                features_df: pd.DataFrame = Featurizer.as_dataframe(features)

                quality = self._pipeline.predict(features_df)[0]

                scores = ClassifierScores(
                    confidence=self._pipeline.predict_proba(features_df).max(),
                    n_characters=len(page),
                    n_tokens=len(tokens),
                    language=language,
                    language_confidence=language_confidence,
                    **features,
                )
                reason = Reason.CLASSIFIER
            else:
                logging.info(
                    "Language '%s' differs from default language '%s'.",
                    language,
                    self._default_language,
                )
                reason = Reason.LANGUAGE
                quality = EMPTY_PAGE_OUTPUT
                scores = default_scores_dict(
                    0,
                    confidence=0.0,
                    n_characters=len(page),
                    language=language,
                    language_confidence=language_confidence,
                )

        return quality, scores, reason

    def _classify_pagexml_with_scores(
        self, pagexml: Page
    ) -> tuple[int, ClassifierScores, Reason]:
        """Classify a Page object with scores."""

        if all(len(line) < SHORT_COLUMN_WIDTH for line in pagexml.lines()):
            logging.warning("Page '%s' has short columns.", pagexml.id)

            quality = 3
            scores = default_scores_dict(
                0, confidence=1.0, n_characters=len(pagexml.get_text())
            )
            reason = Reason.SHORT_COLUMNS
        else:
            quality, scores, reason = self.classify_with_scores(pagexml.get_text())

        return quality, scores, reason

    @staticmethod
    def _is_short(text: str):
        return len(text.strip()) < MINIMUM_PAGE_LENGTH and EMPTY_PAGE_OUTPUT is not None

    @classmethod
    def from_file(cls, pipeline_file: Path, featurizer: Featurizer):
        """Load a pipeline from a file."""
        logging.info("Reading classifier pipeline from file '%s'.", str(pipeline_file))
        return cls(joblib.load(pipeline_file), featurizer)
