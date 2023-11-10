import tempfile
from contextlib import nullcontext as does_not_raise
from pathlib import Path
import pytest
from text_quality.language.classifier import LanguageClassifier
from text_quality.language.fasttext import FastTextLanguageClassifier


@pytest.fixture(scope="session")
def model_file() -> Path:
    return Path(tempfile.gettempdir()) / "lid.176.ftz"


class TestLanguageClassifier:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", ""),
            ("test text", "test text"),
            ("„test text.", "test text"),
            ("„test . . . text.", "test       text"),
        ],
    )
    def test_preprocess(self, text: str, expected: str):
        assert LanguageClassifier.preprocess(text) == expected


@pytest.mark.internet_access
class TestFastTextLanguageClassifier:
    @pytest.mark.parametrize(
        "text,expected_language,expected_confidence",
        [
            ("An English text", "en", 1.0),
            ("Een Nederlandse tekst", "nl", 1.0),
            ("A multiline\nEnglish text", "en", 1.0),
            (
                "A multi-lingual English text\nmet een Nederlandse regel\nand back to English.",
                "en",
                pytest.approx(0.6102203526820273),
            ),
            ("", "", 0.0),
            ("... . !@#!@# \n . --- ---", "", 0.0),
        ],
    )
    def test_classify(
        self, model_file: Path, text: str, expected_language, expected_confidence
    ) -> str:
        classifier = FastTextLanguageClassifier(model_file=model_file)
        assert (
            expected_language,
            expected_confidence,
        ) == classifier.classify(text)

    @pytest.mark.parametrize(
        "model_file,expected_exception",
        [
            (Path("lid.176.ftz"), does_not_raise()),
            (Path("unknown_model"), pytest.raises(ValueError)),
        ],
    )
    def test_download(self, tmp_path: Path, model_file, expected_exception):
        """Download model file."""
        model_path = tmp_path / model_file

        with expected_exception:
            classifier = FastTextLanguageClassifier(model_file=model_path)
            assert ("en", 1.0) == classifier.classify(
                "An English text",
            )
