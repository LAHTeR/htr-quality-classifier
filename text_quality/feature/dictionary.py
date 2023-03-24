from abc import ABC
from abc import abstractmethod
import logging
from pathlib import Path
from typing import List
from typing import Set
from spylls import hunspell
from .tokenizer import Tokenizer


class Dictionary(ABC):
    def __init__(self, dictionary) -> None:
        self._dictionary = dictionary

    @abstractmethod
    def _lookup(self, token: str) -> bool:
        return NotImplemented

    def get_score(self, tokens: List[str]) -> float:
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

    def to_file(
        self,
        filepath: Path,
        sort: bool = True,
        overwrite: bool = False,
        line_separator: str = "\n",
    ):
        if filepath.exists() and not overwrite:
            raise FileExistsError(filepath)

        tokens = sorted(self._dictionary) if sort else self._dictionary
        logging.info(f"Writing {len(tokens)} to file '{filepath}'.")

        with open(filepath, "rt") as f:
            f.write(line_separator.join(tokens))

    @classmethod
    def from_file(cls, filepath: Path):
        with open(filepath, "rt") as f:
            return cls(f.readlines())

    @classmethod
    def generate_from_text(cls, text: str, tokenizer: Tokenizer) -> "TokenDictionary":
        return cls(set(tokenizer.tokenize(text)))


class HunspellDictionary(Dictionary):
    def _lookup(self, token: str) -> bool:
        return self._dictionary.lookup(token)

    @classmethod
    def from_path(path: Path, language: str) -> "HunspellDictionary":
        hunspell.Dictionary.from_files(str(path / language))
