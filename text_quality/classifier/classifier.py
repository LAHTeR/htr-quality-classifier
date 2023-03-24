from pathlib import Path
from joblib import load
from sklearn.base import ClassifierMixin

from text_quality.settings import CLASSIFIER_FEATURES


class Classifier:
    def __init__(self, classifier: ClassifierMixin) -> None:
        if not classifier.feature_names_in_ == CLASSIFIER_FEATURES:
            raise ValueError(f"Classifier features do not match.")

        self._classifier = classifier

    @classmethod
    def from_joblib_dump(cls, filepath: Path):
        return cls(load(filepath))
