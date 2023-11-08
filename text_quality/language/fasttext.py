import logging
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.error import URLError
import fasttext
from .classifier import LanguageClassifier


class FastTextLanguageClassifier(LanguageClassifier):
    """LanguageClassifier implementation using FastText."""

    # TODO: use hash values for the model files to validate downloads
    MODEL_URLS = {
        "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",  # 917kb
        "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",  # 126mb
    }
    """URLs for the FastText language models for automatic download."""
    _MODEL_FILE_MD5SUMS = {}

    _LABEL_PREFIX = "__label__"
    """The classifier always returns labels with this prefix; will be removed."""

    def classify(
        self, text: str, *, model_file: Path = Path("lid.176.ftz"), download=True
    ) -> tuple[str, float]:
        """Classify a text string.

        If the model file is not found locally, it will be downloaded automatically.

        Args:
            text: The text to classify.
            model_file: the local path to the model file.
            download: whether to download the model file if it is not found locally.
        Returns:
            A tuple with the language and the confidence.
        Raises:
            FileNotFoundError: if the model file is not found locally and download=False.
            ValueError: if the model file is not found locally and no download location is known.
            RuntimeError: if the model file is not found locally and download fails.
        """
        if model_file.is_file():
            model = fasttext.load_model(str(model_file))
        elif download:
            try:
                self._download_model(model_file.name, target=model_file)
            except KeyError as e:
                raise ValueError(
                    f"Model file '{model_file}' not found, and no download location known for '{model_file.name}'."
                ) from e
            except URLError as e:
                raise RuntimeError(
                    f"Model file '{model_file}' not found, and download failed."
                ) from e
            return self.classify(text, model_file=model_file, download=False)
        else:
            raise FileNotFoundError(f"Model file '{model_file}' not found.")

        labels, confidences = model.predict(
            [LanguageClassifier.preprocess(line) for line in text.split("\n")]
        )
        language, confidence = FastTextLanguageClassifier._aggregate_results(
            labels, confidences
        )
        return language.removeprefix(self._LABEL_PREFIX), confidence

    def _download_model(self, model: str, target: Path):
        url = self.MODEL_URLS[model]

        target.parent.mkdir(parents=True, exist_ok=True)

        logging.info("Downloading from '%s', storing at '%s' .", model, url)
        with urllib.request.urlopen(url) as response, target.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)

    @staticmethod
    def _aggregate_results(labels: list[str], confidences: list) -> tuple[str, float]:
        if len(labels) != len(confidences):
            raise ValueError(
                f"Labels and confidences must have the same length, but were {len(labels)} and {len(confidences)}."
            )

        total_confidence = Counter()
        for _labels, _confidences in zip(labels, confidences):
            # Iterate over labels and confidences per input line (corresponds to k, defaults to 1)
            for label, confidence in zip(_labels, _confidences):
                total_confidence[label] += confidence
        label, total = total_confidence.most_common()[0]
        return label, total / sum(total_confidence.values())
