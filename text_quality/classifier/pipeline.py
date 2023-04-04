import logging
from pathlib import Path
import joblib
import pandas as pd
import sklearn.pipeline
from ..feature.featurize import Featurizer


class Pipeline:
    def __init__(
        self, pipeline: sklearn.pipeline.Pipeline, featurizer: Featurizer
    ) -> None:
        self._pipeline = pipeline
        self._featurizer = featurizer

    def classify(self, text) -> int:
        """Single instance classification"""
        features: pd.DataFrame = self._featurizer.featurize_as_dataframe(text)
        return self._pipeline.predict(features)[0]

    @classmethod
    def from_file(cls, pipeline_file: Path, featurizer: Featurizer):
        logging.info(f"Reading classifier pipeline from file '{str(pipeline_file)}'.")
        return cls(joblib.load(pipeline_file), featurizer)
