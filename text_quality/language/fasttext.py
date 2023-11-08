import logging
import shutil
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.error import URLError
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

    def classify(
        self,
        text: str,
        *,
        model_file: Path = _DEFAULT_MODEL_PATH,
        download=True,
        threshold: float = 0.5,
    ) -> tuple[str, float]:
        """Classify a text string.

        If the model file is not found locally, it will be downloaded automatically.

        There is currently no check in place to validate an existing model file.

        Args:
            text (str): The text to classify.
            model_file (Path): the local path to the model file.
            download (bool): whether to download the model file if it is not found locally.
            threshold (float): filter out predictions for lines where the classifier confidence is below this threshold.
        Returns:
            A tuple[str, float] with the language code (e.g. "nl") and the confidence.
        Raises:
            FileNotFoundError: if the model file is not found locally and download=False.
            ValueError: if the model file is not found locally and no download location is known.
            RuntimeError: if the model file is not found locally and download fails.
        """
        if model_file.is_file():
            model = fasttext.load_model(str(model_file))
        elif download:
            try:
                FastTextLanguageClassifier._download_model(
                    model_file.name, target=model_file
                )
            except KeyError as e:
                raise ValueError(
                    f"Model file '{model_file}' not found, and no download location known for '{model_file.name}'."
                ) from e
            except URLError as e:
                raise RuntimeError(
                    f"Model file '{model_file}' not found, and download failed."
                ) from e

            # Retry with downloaded model file, do not retry to download
            return self.classify(
                text, model_file=model_file, download=False, threshold=threshold
            )
        else:
            raise FileNotFoundError(f"Model file '{model_file}' not found.")

        lines: list[str] = [
            LanguageClassifier.preprocess(line) for line in text.split("\n")
        ]
        line_labels, line_confidences = model.predict(lines, threshold=threshold)
        language, confidence = FastTextLanguageClassifier._aggregate_lines(
            line_labels, line_confidences
        )

        return language.removeprefix(self._LABEL_PREFIX), confidence

    @staticmethod
    def _download_model(model: str, target: Path):
        url = FastTextLanguageClassifier.MODEL_URLS[model]

        target.parent.mkdir(parents=True, exist_ok=True)

        logging.info("Downloading from '%s', storing at '%s' .", model, url)
        with urllib.request.urlopen(url) as response, target.open("wb") as out_file:
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
