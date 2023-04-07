from abc import ABC
from abc import abstractmethod
from typing import List


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        return NotImplemented


class NautilusOcrTokenizer(Tokenizer):
    _HYPHENS = {"-", "⸗", "="}

    def tokenize(self, text: str) -> List[str]:
        """`Nautilus-OCR tokenizer <https://github.com/natliblux/nautilusocr/blob/2d4d59c45466b5cc8c9897798bd8b205a7f0c02c/src/epr/features_epr.py#L84>`_"""

        tokens = []

        new_token = ""
        for c in text:
            if c == " " and len(new_token) > 0:
                tokens.append(new_token)
                new_token = ""
            elif c == "\n" and len(new_token) > 0:
                if new_token[-1] in self._HYPHENS:
                    new_token = new_token[:-1]
                else:
                    tokens.append(new_token)
                    new_token = ""
            else:
                new_token += c
        if len(new_token) > 0:
            tokens.append(new_token)

        for i, token in enumerate(tokens):
            if not token[-1].isalpha():
                tokens[i] = token[:-1]
            if not token[0].isalpha():
                tokens[i] = token[1:]

        return tokens
