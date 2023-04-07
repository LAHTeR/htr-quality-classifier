import logging
from functools import lru_cache
from pathlib import Path
from typing import List
from typing import Optional
from ...settings import ENCODING
from ...settings import LINE_SEPARATOR
from ...settings import Q_GRAM_LENGTH
from ...settings import Q_GRAMS_GAMMA
from .scorer import Scorer


class QGram(Scorer):
    def __init__(self, qgrams: List[str]) -> None:

        self._lang_qgrams = qgrams
        self._qgram_set = set(qgrams)

    @lru_cache
    def get_rank(self, qgram: str) -> Optional[int]:
        if qgram not in self._qgram_set:
            return None
        return self._lang_qgrams.index(qgram)

    @lru_cache(maxsize=1024)
    def _get_ngram_score(self, ngram: str) -> float:
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(self._lang_qgrams)):
            if ngram == self._lang_qgrams[i]:
                return 1 - (1 / len(self._lang_qgrams) * i)
        raise AssertionError()

    def _get_ngram_scores(self, ngrams: List[str]) -> float:
        """
        `See Nautilus-OCR <https://github.com/natliblux/nautilusocr/blob/2d4d59c45466b5cc8c9897798bd8b205a7f0c02c/src/epr/features_epr.py#L51>`_
        """
        if len(ngrams) == 0:
            return 0

        score = 0
        for ngram in ngrams:
            if ngram in self._qgram_set:
                score += self._get_ngram_score(ngram)

        score = score / len(ngrams)
        return score

    def score(self, tokens: List[str]) -> float:
        return self._get_ngram_scores(QGram._get_qgrams(tokens))

    def to_file(self, filepath: Path):
        if filepath.exists():
            raise FileExistsError(filepath)
        with open(filepath, "wt", encoding=ENCODING) as f:
            f.write(LINE_SEPARATOR.join(self._lang_qgrams))

    @staticmethod
    def _get_qgrams(tokens: List[str]):
        """
        `See Nautilus-OCR <https://github.com/natliblux/nautilusocr/blob/2d4d59c45466b5cc8c9897798bd8b205a7f0c02c/src/epr/features_epr.py#L51>`_
        """

        q_grams = []
        for token in tokens:
            token_list = list(token)
            # pylint: disable=consider-using-enumerate
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
            "Reading character q-grams from file '%s', with gamma=%d.",
            str(filepath),
            gamma,
        )
        q_grams = []
        with open(filepath, "rt", encoding=ENCODING) as f:
            for line in f:
                q_grams.append(line.strip())
                if gamma and len(q_grams) >= gamma:
                    logging.info(
                        "Stopping reading q-grams list, %d q-grams read.", len(q_grams)
                    )
                    break
        return cls(q_grams)
