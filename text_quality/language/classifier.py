import abc
import string


class LanguageClassifier(abc.ABC):
    """Abstract class for implementing different language classifiers."""

    REMOVE_CHARACTERS = string.punctuation + "„"

    @abc.abstractmethod
    def classify(self, text: str) -> tuple[str, float]:
        """Classify a text string.

        Args:
            text: The text to classify.

        Returns:
            A tuple with the language and the confidence.
        """
        return NotImplemented

    @staticmethod
    def preprocess(text: str) -> str:
        for c in LanguageClassifier.REMOVE_CHARACTERS:
            text = text.replace(c, " ")  # noqa: self-cls-assignment
        return text.strip()
