"""Classification pipeline."""

import logging
from pathlib import Path
from typing import List
from typing import TypedDict
from typing import Union
import joblib
import pandas as pd
import sklearn.pipeline
from ..feature.featurize import Featurizer
from ..feature.featurize import Scorers
from ..page.page import Page
from ..settings import EMPTY_PAGE_OUTPUT
from ..settings import MINIMUM_PAGE_LENGTH
from ..settings import SHORT_COLUMN_WIDTH


ClassifierScores = TypedDict(
    "ClassifierScores",
    {"confidence": float, "n_characters": int, "n_tokens": int}
    | {score: float for score in Scorers.__annotations__.keys()},
)
"""Container class for the scores returned by the classifier."""


def default_scores_dict(default_value, **kwargs) -> ClassifierScores:
    """Generate a ClassifierScores dict with default values."""

    return ClassifierScores(
        {field: default_value for field in ClassifierScores.__annotations__.keys()}
        | kwargs
    )


class Pipeline:
    """A wrapper around an sklearn pipeline that adds a featurizer."""

    def __init__(
        self, pipeline: sklearn.pipeline.Pipeline, featurizer: Featurizer
    ) -> None:
        self._pipeline = pipeline
        self._featurizer = featurizer

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
            features, _ = self._featurizer.featurize_as_dataframe(page)
            quality = self._pipeline.predict(features)[0]

        return quality

    def _classify_pagexml(self, pagexml: Page) -> int:
        """Classify a Page object."""

        if all(len(line) < SHORT_COLUMN_WIDTH for line in pagexml.lines()):
            logging.warning("Page '%s' has short columns.", pagexml.id)
            quality = 3
        else:
            quality = self.classify(pagexml.get_text())

        return quality

    def classify_with_scores(
        self, page: Union[Page, str]
    ) -> tuple[int, ClassifierScores]:
        """Single instance classification with scores."""

        if isinstance(page, Page):
            quality, scores = self._classify_pagexml_with_scores(page)
        elif self._is_short(page):
            logging.debug(
                "Skipping short text: '%s' (%d characters).", page, len(page.strip())
            )
            confidence = 1.0
            quality = EMPTY_PAGE_OUTPUT
            scores = default_scores_dict(0, confidence=1.0, n_characters=len(page))
        else:
            features, tokens = self._featurizer.featurize(page)
            features_df: pd.DataFrame = Featurizer.as_dataframe(features)

            quality = self._pipeline.predict(features_df)[0]

            confidence: float = self._pipeline.predict_proba(features_df).max()
            scores = ClassifierScores(
                confidence=confidence,
                n_characters=len(page),
                n_tokens=len(tokens),
                **features
            )

        return quality, scores

    def _classify_pagexml_with_scores(
        self, pagexml: Page
    ) -> tuple[int, ClassifierScores]:
        """Classify a Page object with scores."""

        if all(len(line) < SHORT_COLUMN_WIDTH for line in pagexml.lines()):
            logging.warning("Page '%s' has short columns.", pagexml.id)

            quality = 3
            scores = default_scores_dict(
                0, confidence=1.0, n_characters=len(pagexml.get_text())
            )
        else:
            quality, scores = self.classify_with_scores(pagexml.get_text())

        return quality, scores

    @staticmethod
    def _is_short(text: str):
        return len(text.strip()) < MINIMUM_PAGE_LENGTH and EMPTY_PAGE_OUTPUT is not None

    @classmethod
    def from_file(cls, pipeline_file: Path, featurizer: Featurizer):
        """Load a pipeline from a file."""
        logging.info("Reading classifier pipeline from file '%s'.", str(pipeline_file))
        return cls(joblib.load(pipeline_file), featurizer)
