from abc import ABC, abstractmethod
from ..model import Model
from typing import List

class Decomposer(ABC):
    def __init__(self, model: Model):
        self.origin_model = model

    @abstractmethod
    def decompose() -> List[Model]:
        pass



