import pytest
from text_quality.feature.scorer.garbage import GarbageDetector


@pytest.fixture
def garbage_detector():
    return GarbageDetector()


class TestGarbageDetector:
    @pytest.mark.parametrize(
        "tokens, expected",
        [
            ([], 0),
            (["token1", "token2", "token3"], 0),
            (["a" * 22], 1),
            (["î" * 5], 1),
            (["k" * 7], 1),
            (["a" * 22, "token", "token"], 1 / 3),
        ],
    )
    def test_score(self, garbage_detector, tokens, expected):
        assert garbage_detector.score(tokens) == pytest.approx(expected)
