from abc import ABC
from abc import abstractmethod
from typing import List


class Scorer(ABC):
    """Abstract class for scorers to compute feature values"""

    @abstractmethod
    def score(self, tokens: List[str]) -> float:
        return NotImplemented
