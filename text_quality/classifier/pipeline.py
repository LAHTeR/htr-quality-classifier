import logging
from pathlib import Path
from typing import List
from typing import TypedDict
import joblib
import pandas as pd
import sklearn.pipeline
from ..feature.featurize import Featurizer
from ..feature.featurize import Scorers


ClassifierScores = TypedDict(
    "ClassifierScores",
    {"confidence": float, "n_characters": int, "n_tokens": int}
    | {score: float for score in Scorers.__annotations__.keys()},
)


class Pipeline:
    def __init__(
        self, pipeline: sklearn.pipeline.Pipeline, featurizer: Featurizer
    ) -> None:
        self._pipeline = pipeline
        self._featurizer = featurizer

    @property
    def features(self) -> List[str]:
        return list(self._pipeline.feature_names_in_)

    def classify(self, text) -> int:
        """Single instance classification"""
        features, _ = self._featurizer.featurize_as_dataframe(text)
        return self._pipeline.predict(features)[0]

    def classify_with_scores(self, text) -> tuple[int, ClassifierScores]:
        features, tokens = self._featurizer.featurize(text)
        features_df: pd.DataFrame = Featurizer.as_dataframe(features)
        confidence: float = self._pipeline.predict_proba(features_df).max()

        return self._pipeline.predict(features_df)[0], ClassifierScores(
            confidence=confidence,
            n_characters=len(text),
            n_tokens=len(tokens),
            **features,
        )

    @classmethod
    def from_file(cls, pipeline_file: Path, featurizer: Featurizer):
        logging.info("Reading classifier pipeline from file '%s'.", str(pipeline_file))
        return cls(joblib.load(pipeline_file), featurizer)
