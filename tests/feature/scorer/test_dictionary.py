import tempfile
from pathlib import Path
import pytest
from text_quality.classifier.settings import ENCODING
from text_quality.classifier.settings import HUNSPELL_DIR
from text_quality.classifier.settings import HUNSPELL_LANGUAGE
from text_quality.feature.scorer.dictionary import HunspellDictionary
from text_quality.feature.scorer.dictionary import TokenDictionary


@pytest.fixture
def token_dictionary():
    return TokenDictionary(["token"])


@pytest.fixture
def hunspell_dictionary():
    return HunspellDictionary.from_path(HUNSPELL_DIR, HUNSPELL_LANGUAGE)


@pytest.fixture
def token_file(tmp_path):
    tokens = ["token1", "token2"]
    with tempfile.NamedTemporaryFile(dir=tmp_path, delete=False) as tokens_file:
        for token in tokens:
            tokens_file.write((token + "\n").encode(ENCODING))
    return tokens_file.name


class TestTokenDictionary:
    # pylint: disable=protected-access

    @pytest.mark.parametrize(
        "token,expected", [("token", True), ("test", False), ("", False)]
    )
    def test_lookup(self, token_dictionary, token, expected):
        assert token_dictionary._lookup(token) == expected

    def test_from_file(self, token_file):
        assert TokenDictionary.from_file(token_file)._dictionary == {"token1", "token2"}

    def test_to_from_file(self, token_dictionary, tmp_path):
        dict_file = Path(tmp_path) / "dictionary"

        token_dictionary.to_file(Path(dict_file))

        assert (
            TokenDictionary.from_file(dict_file)._dictionary
            == token_dictionary._dictionary
        )

    @pytest.mark.parametrize(
        "tokens, expected",
        [([], 0), (["token"], 1), (["token", "token2"], 0.4545), (["Token"], 0)],
    )
    def test_score(self, token_dictionary, tokens, expected):
        assert token_dictionary.score(tokens) == pytest.approx(expected, 0.001)


class TestHunspellDictionary:
    # pylint: disable=protected-access

    def test_from_path(self, hunspell_dictionary):
        assert len(hunspell_dictionary._dictionary.dic.index) == 177280
        assert len(hunspell_dictionary._dictionary.dic.words) == 180960

    @pytest.mark.parametrize(
        "token, expected",
        [("", False), ("Nederland", True), ("asdasdasdasdasdasdasds", False)],
    )
    def test_lookup(self, hunspell_dictionary, token, expected):
        assert hunspell_dictionary._lookup(token) == expected

    @pytest.mark.parametrize(
        "text, expected", [("", 0), ("een test tekst", 0.8571428571428571)]
    )
    def test_score(self, hunspell_dictionary, text, expected):
        assert hunspell_dictionary.score(text) == pytest.approx(expected)
