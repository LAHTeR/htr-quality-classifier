"""Tests for the tokenizer module."""
import pytest
from text_quality.feature.tokenizer import NautilusOcrTokenizer


class TestNautilusOcrTokenizer:
    tokenizer = NautilusOcrTokenizer()

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", []),
            ("test token", ["test", "token"]),
            ("test-token", ["test-token"]),
            ("test 1", ["test", ""]),
            ("test 123a", ["test", "23a"]),
        ],
    )
    def test_tokenize(self, text, expected):
        assert self.tokenizer.tokenize(text) == expected
