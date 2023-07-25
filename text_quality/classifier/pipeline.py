"""Classification pipeline."""

import logging
from pathlib import Path
from typing import List
from typing import TypedDict
import joblib
import pandas as pd
import sklearn.pipeline
from ..feature.featurize import Featurizer
from ..feature.featurize import Scorers
from ..settings import EMPTY_PAGE_OUTPUT
from ..settings import MINIMUM_PAGE_LENGTH


ClassifierScores = TypedDict(
    "ClassifierScores",
    {"confidence": float, "n_characters": int, "n_tokens": int}
    | {score: float for score in Scorers.__annotations__.keys()},
)
"""Container class for the scores returned by the classifier."""


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

    def classify(self, text) -> int:
        """Single instance classification."""

        if (len(text.strip()) >= MINIMUM_PAGE_LENGTH) or (EMPTY_PAGE_OUTPUT is None):
            features, _ = self._featurizer.featurize_as_dataframe(text)
            result = self._pipeline.predict(features)[0]
        else:
            result = EMPTY_PAGE_OUTPUT

        return result

    def classify_with_scores(self, text) -> tuple[int, ClassifierScores]:
        """Single instance classification with scores."""

        if (len(text.strip()) >= MINIMUM_PAGE_LENGTH) or (EMPTY_PAGE_OUTPUT is None):
            features, tokens = self._featurizer.featurize(text)
            features_df: pd.DataFrame = Featurizer.as_dataframe(features)

            confidence: float = self._pipeline.predict_proba(features_df).max()
            result = self._pipeline.predict(features_df)[0]
        else:
            confidence = 1.0
            result = EMPTY_PAGE_OUTPUT

        return result, ClassifierScores(
            confidence=confidence,
            n_characters=len(text),
            n_tokens=len(tokens),
            **features,
        )

    @classmethod
    def from_file(cls, pipeline_file: Path, featurizer: Featurizer):
        """Load a pipeline from a file."""
        logging.info("Reading classifier pipeline from file '%s'.", str(pipeline_file))
        return cls(joblib.load(pipeline_file), featurizer)
