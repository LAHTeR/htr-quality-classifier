from collections import Counter
from functools import lru_cache
from typing import List
from typing import Optional

from .tokenizer import Tokenizer


class QGram:
    _Q_GRAM_LENGTH: int = 3

    def __init__(self, qgrams: List[str], tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

        self._qgrams = qgrams
        self._qgram_set = set(qgrams)

        if len(self._qgram_set) != len(self._qgrams):
            raise ValueError("List must contain unique q-grams.")

    @lru_cache
    def get_rank(self, qgram: str) -> Optional[int]:
        if qgram in self._qgram_set:
            return self._qgrams.index(qgram)
        else:
            return None

    def _get_ngram_score(self, ngrams: List[str]) -> float:
        """Copied from features_epr.py"""
        # TODO: this is very slow; pre-select q-grams that occur in self._qgrams, use rank
        # TODO: check if this corresponds to equation 10 in the paper?

        if len(ngrams) == 0:
            return 0

        score = 0
        for ngram in ngrams:
            for i in range(0, len(self._qgrams)):
                if ngram == self._qgrams[i]:
                    score += 1 - (1 / len(self._qgrams) * i)
                    break

        score = score / len(ngrams)
        return score

    def score(self, text: str) -> float:
        return self._get_ngram_score(QGram.get_qgrams(self._tokenizer.tokenize(text)))

    @staticmethod
    def get_qgrams(tokens: List[str]):
        """Copied, adapted from features_epr.py"""
        # TODO: re-implement

        q_grams = list()
        for token in tokens:
            token_list = list(token)
            for i in range(0, len(token_list)):
                if not token[i].isalpha():
                    token_list[i] = " "
            modified_token = "".join(token_list)
            splits = modified_token.split(" ")
            for split in splits:
                if split != "":
                    for i in range(0, len(split) - QGram._Q_GRAM_LENGTH + 1):
                        q_grams.append(split[i : i + QGram._Q_GRAM_LENGTH].lower())
        return q_grams

    @classmethod
    def from_text(cls, text: str, tokenizer: Tokenizer, max_rank: int):
        return cls(
            [
                qgram
                for qgram, _count in Counter(
                    cls.get_qgrams(tokenizer.tokenize(text))
                ).most_common(max_rank)
            ],
            tokenizer,
        )

