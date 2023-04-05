import logging
from abc import abstractmethod
from pathlib import Path
from typing import List
from typing import Set
from spylls import hunspell
from text_quality.settings import LINE_SEPARATOR
from .scorer import Scorer


class Dictionary(Scorer):
    def __init__(self, dictionary) -> None:
        self._dictionary = dictionary

    @abstractmethod
    def _lookup(self, token: str) -> bool:
        return NotImplemented

    def score(self, tokens: List[str]) -> float:
        if any(len(token) > 0 for token in tokens):
            matched_count = 0
            total_count = 0

            for token in tokens:
                total_count += len(token)

                # TODO: lowercase token?
                matched_count += self._lookup(token) * len(token)

            return matched_count / total_count
        else:
            # empty input
            return 0.0


class TokenDictionary(Dictionary):
    def __init__(self, dictionary) -> None:
        super().__init__(set(dictionary))

    def _lookup(self, token: str) -> bool:
        return token in self._dictionary

    def to_file(self, filepath: Path, sort: bool = True, overwrite: bool = False):
        if filepath.exists() and not overwrite:
            raise FileExistsError(filepath)

        tokens = sorted(self._dictionary) if sort else self._dictionary
        logging.info(f"Writing {len(tokens)} to file '{filepath}'.")

        with open(filepath, "wt") as f:
            f.write(LINE_SEPARATOR.join(tokens))

    @classmethod
    def from_file(cls, filepath: Path):
        logging.info(f"Reading token dictionary from file '{str(filepath)}'.")
        with open(filepath, "rt") as f:
            tokens = [line.strip() for line in f if not line.strip().startswith("#")]
        return cls(tokens)


class HunspellDictionary(Dictionary):
    def _lookup(self, token: str) -> bool:
        return len(token.strip()) > 0 and self._dictionary.lookup(token)

    @classmethod
    def from_path(cls, path: Path, language: str) -> "HunspellDictionary":
        logging.info(
            f"Reading Hunspell dictionary '{language}' in directory '{str(path)}'"
        )
        return cls(hunspell.Dictionary.from_files(str(path / language)))
