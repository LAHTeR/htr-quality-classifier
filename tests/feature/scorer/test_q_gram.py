from pathlib import Path
import pytest

from text_quality.feature.scorer.q_gram import QGram


@pytest.fixture
def q_gram():
    return QGram(["abc", "def"])


class TestQGram:
    @pytest.mark.parametrize(
        "tokens, expected",
        [
            ([], 0),
            (["token"], 0),
            (["abcdef"], 0.375),
            (["abcdef", "test"], 0.25),
            (["abc", "def"], 0.75),
            (["abcd", "ef"], 0.5),
        ],
    )
    def test_score(self, q_gram, tokens, expected):
        assert q_gram.score(tokens) == pytest.approx(expected)

    def test_to_from_file(self, tmp_path, q_gram):
        # pylint: disable=W0212

        qgram_file = Path(tmp_path) / "qgrams.txt"
        q_gram.to_file(qgram_file)

        assert QGram.from_file(qgram_file)._lang_qgrams == ["abc", "def"]
