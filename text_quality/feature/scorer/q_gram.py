import logging
from functools import lru_cache
from pathlib import Path
from typing import List
from typing import Optional
from ...settings import Q_GRAM_LENGTH
from ...settings import Q_GRAMS_GAMMA
from .scorer import Scorer


class QGram(Scorer):
    def __init__(self, qgrams: List[str]) -> None:

        self._lang_qgrams = qgrams
        self._qgram_set = set(qgrams)

    @lru_cache
    def get_rank(self, qgram: str) -> Optional[int]:
        if qgram in self._qgram_set:
            return self._lang_qgrams.index(qgram)
        else:
            return None

    @lru_cache(maxsize=None)
    def _get_ngram_score(self, ngram: str) -> float:
        for i in range(0, len(self._lang_qgrams)):
            if ngram == self._lang_qgrams[i]:
                return 1 - (1 / len(self._lang_qgrams) * i)
        raise AssertionError()

    def _get_ngram_scores(self, ngrams: List[str]) -> float:
        """Copied from features_epr.py"""
        # TODO: this is very slow; pre-select q-grams that occur in self._qgrams, use rank
        # TODO: check if this corresponds to equation 10 in the paper?

        if len(ngrams) == 0:
            return 0

        score = 0
        for ngram in ngrams:
            if ngram in self._qgram_set:
                score += self._get_ngram_score(ngram)
            # for i in range(0, len(self._lang_qgrams)):
            #     if ngram == self._lang_qgrams[i]:
            #         score += 1 - (1 / len(self._lang_qgrams) * i)
            #         break

        score = score / len(ngrams)
        return score

    def score(self, tokens: List[str]) -> float:
        return self._get_ngram_scores(QGram.get_qgrams(tokens))

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
                    for i in range(0, len(split) - Q_GRAM_LENGTH + 1):
                        q_grams.append(split[i : i + Q_GRAM_LENGTH].lower())
        return q_grams

    @classmethod
    def from_file(cls, filepath: Path, gamma: int = Q_GRAMS_GAMMA):
        logging.info(
            f"Reading character q-grams from file '{str(filepath)}', with gamma={gamma}."
        )
        q_grams = []
        with open(filepath, "rt") as f:
            for line in f:
                # TODO: call decode(encoding) for each line?
                q_grams.append(line.strip())
                if gamma and len(q_grams) >= gamma:
                    logging.info(
                        f"Stopping reading q-grams list, {len(q_grams)} q-grams read."
                    )
                    break
        return cls(q_grams)
