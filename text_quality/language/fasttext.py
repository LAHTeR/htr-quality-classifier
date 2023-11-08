import logging
import shutil
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
import fasttext
from numpy.typing import ArrayLike
from .classifier import LanguageClassifier


class FastTextLanguageClassifier(LanguageClassifier):
    """LanguageClassifier implementation using FastText."""

    MODEL_URLS: dict[str, str] = {
        "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",  # 917kb
        "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",  # 126mb
    }
    """URLs for the FastText language models for automatic download."""

    _DEFAULT_MODEL_PATH: Path = Path(tempfile.gettempdir()) / "lid.176.ftz"
    """Default location for the FastText language model file."""

    _LABEL_PREFIX: str = "__label__"
    """The classifier always returns labels with this prefix; removed before returning it."""

    def __init__(
        self, *, model_file: Path = _DEFAULT_MODEL_PATH, line_threshold: float = 0.5
    ):
        """Initialize the classifier.

        If the model file is not found locally, it will be downloaded automatically.
        There is currently no check in place to validate an existing model file.

        Args:
            model_file (Path): the local path to the model file.
            download (bool): whether to download the model file if it is not found locally.
            line_threshold (float): filter out predictions for lines where the classifier confidence is below this threshold.

        Raises:
            ValueError: if the model file is not found locally and no download location is known.
            RuntimeError: if the model file is not found locally and download fails.
        """
        super().__init__()

        if not model_file.exists():
            self._download_model(model_file)
        self._model = fasttext.load_model(str(model_file))

        self._line_threshold = line_threshold

    def classify(self, text: str) -> tuple[str, float]:
        """Classify a text string.

        Args:
            text (str): The text to classify.
        Returns:
            A tuple[str, float] with the language code (e.g. "nl") and the confidence.
        """

        lines: list[str] = [
            LanguageClassifier.preprocess(line) for line in text.split("\n")
        ]
        line_labels, line_confidences = self._model.predict(
            lines, threshold=self._line_threshold
        )
        language, confidence = FastTextLanguageClassifier._aggregate_lines(
            line_labels, line_confidences
        )

        return language.removeprefix(self._LABEL_PREFIX), confidence

    @staticmethod
    def _download_model(model_file: Path):
        try:
            url = FastTextLanguageClassifier.MODEL_URLS[model_file.name]
        except KeyError as e:
            raise ValueError(
                f"No download location known for '{model_file.name}'."
            ) from e

        model_file.parent.mkdir(parents=True, exist_ok=True)

        logging.info("Downloading from '%s', storing at '%s' .", model_file, url)
        with urllib.request.urlopen(url) as response, model_file.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)

    @staticmethod
    def _aggregate_lines(
        line_labels: list[list[str]], line_confidences: list[ArrayLike]
    ) -> tuple[str, float]:
        """Aggregate the results per line from the classifier.

        Args:
            line_labels (list[list[str]]): the labels returned by the classifier; one list of labels for each input line
            line_confidences (list[ArrayLike]): the confidences returned by the classifier; one array for each input line
        Returns:
            A tuple[str, float] with the language code (e.g. "nl") and the confidence.
                ("None", 0.0) if no input line could be classified.
        """
        if len(line_labels) != len(line_confidences):
            raise ValueError(
                f"Labels and confidences must have the same length, but were {len(line_labels)} and {len(line_confidences)}."
            )

        if any(labels for labels in line_labels):
            total_confidences = Counter()
            for labels, confidences in zip(line_labels, line_confidences):
                # Iterate over labels and confidences per input line
                # Number of entries per line corresponds to k parameter in model.predict(), defaults to 1
                for label, confidence in zip(labels, confidences):
                    total_confidences[label] += confidence
            label, total_confidence = total_confidences.most_common()[0]
            confidence = total_confidence / sum(total_confidences.values())
        else:
            # Classifier could not determine language for any line in the input
            label = str(None)
            confidence = 0.0

        return label, confidence
