import tempfile
from contextlib import nullcontext as does_not_raise
from pathlib import Path
import pytest
from text_quality.language.fasttext import FastTextLanguageClassifier


@pytest.fixture(scope="session")
def model_file() -> Path:
    return Path(tempfile.gettempdir()) / "lid.176.ftz"


class TestFastTextLanguageClassifier:
    @pytest.mark.parametrize(
        "text,expected_language,expected_confidence",
        [
            ("An English text", "en", 1.0),
            ("Een Nederlandse tekst", "nl", 1.0),
            ("A multiline\nEnglish text", "en", 1.0),
            (
                "A multi-lingual English text\nmet een Nederlands regel\nand back to English.",
                "en",
                pytest.approx(0.6102, abs=1e-3),
            ),
        ],
    )
    def test_classify(
        self, model_file: Path, text: str, expected_language, expected_confidence
    ) -> str:
        assert (
            expected_language,
            expected_confidence,
        ) == FastTextLanguageClassifier().classify(text, model_file=model_file)

    @pytest.mark.parametrize(
        "model_file,download,expected_exception",
        [
            (Path("lid.176.ftz"), True, does_not_raise()),
            (Path("unknown_model"), True, pytest.raises(ValueError)),
            (Path("lid.176.ftz"), False, pytest.raises(FileNotFoundError)),
        ],
    )
    def test_download(self, tmp_path: Path, model_file, download, expected_exception):
        """Download model file."""
        model_path = tmp_path / model_file
        assert not model_path.exists()

        with pytest.raises(FileNotFoundError):
            FastTextLanguageClassifier().classify(
                "An English text", model_file=model_path, download=False
            )

        with expected_exception:
            assert FastTextLanguageClassifier().classify(
                "An English text", model_file=model_path, download=download
            ) == ("en", 1.0)
