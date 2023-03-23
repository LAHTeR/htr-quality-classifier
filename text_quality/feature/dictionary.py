from pathlib import Path
from typing import Collection
from typing import List
from typing import Set
from spylls import hunspell
from .tokenizer import Tokenizer


class Dictionary:
    def __init__(self, words: Collection[str]) -> None:
        self._dictionary: Set[str] = set(words)

    def _lookup(self, token: str) -> bool:
        return token in self._dictionary

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

    @classmethod
    def generate_from_tokens(cls, tokens: List[str]) -> "Dictionary":
        return cls(set(tokens))

    @classmethod
    def generate_from_text(cls, text: str, tokenizer: Tokenizer) -> "Dictionary":
        # TODO tokenizer type hint
        return cls(set(tokenizer.tokenize(text)))


class HunspellDictionary(Dictionary):
    def __init__(self, hunspell: hunspell.Dictionary) -> None:
        self._dictionary = hunspell

    def _lookup(self, token: str) -> bool:
        return self._dictionary.lookup(token)

    @classmethod
    def from_path(path: Path, language: str):
        hunspell.Dictionary.from_files(str(path / language))
