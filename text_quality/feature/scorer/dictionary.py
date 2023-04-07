import logging
from abc import abstractmethod
from pathlib import Path
from typing import List
from spylls import hunspell
from ...settings import ENCODING
from ...settings import LINE_SEPARATOR
from .scorer import Scorer


class Dictionary(Scorer):
    def __init__(self, dictionary) -> None:
        self._dictionary = dictionary

    @abstractmethod
    def _lookup(self, token: str) -> bool:
        return NotImplemented

    def score(self, tokens: List[str]) -> float:
        """
        `See Nautilus-OCR <https://github.com/natliblux/nautilusocr/blob/2d4d59c45466b5cc8c9897798bd8b205a7f0c02c/src/epr/features_epr.py#L129>`_
        """
        if not any(len(token) > 0 for token in tokens):
            # empty input
            return 0.0

        matched_count = 0
        total_count = 0

        for token in tokens:
            total_count += len(token)

            # TODO: lowercase token?
            matched_count += self._lookup(token) * len(token)

        return matched_count / total_count


class TokenDictionary(Dictionary):
    def __init__(self, dictionary) -> None:
        super().__init__(set(dictionary))

    def _lookup(self, token: str) -> bool:
        return token in self._dictionary

    def to_file(self, filepath: Path, sort: bool = True, overwrite: bool = False):
        if filepath.exists() and not overwrite:
            raise FileExistsError(filepath)

        tokens = sorted(self._dictionary) if sort else self._dictionary
        logging.info("Writing %d tokens to file '%s'.", len(tokens), filepath)

        with open(filepath, "wt", encoding=ENCODING) as f:
            f.write(LINE_SEPARATOR.join(tokens))

    @classmethod
    def from_file(cls, filepath: Path):
        logging.info("Reading token dictionary from file '%s'.", str(filepath))
        with open(filepath, "rt", encoding=ENCODING) as f:
            tokens = [line.strip() for line in f if not line.strip().startswith("#")]
        return cls(tokens)


class HunspellDictionary(Dictionary):
    def _lookup(self, token: str) -> bool:
        return len(token.strip()) > 0 and self._dictionary.lookup(token)

    @classmethod
    def from_path(cls, path: Path, language: str) -> "HunspellDictionary":
        logging.info(
            "Reading Hunspell dictionary '%s' in directory '%s'", language, str(path)
        )
        return cls(hunspell.Dictionary.from_files(str(path / language)))
