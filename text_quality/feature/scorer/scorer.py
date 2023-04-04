from abc import ABC
from typing import List


class Scorer(ABC):
    """Abstract class for scorers to compute feature values"""

    def score(self, tokens: List[str]) -> float:
        return NotImplemented
