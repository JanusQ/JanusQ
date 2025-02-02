from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Generic, TypeVar
from qiskit import QuantumCircuit
from ...options import CircuitOption, ModelOption
import numpy as np

T = TypeVar("T", bound=CircuitOption)

class PennylaneCircuit(ABC, Generic[T]):
    def __init__(self, circuit_option: T, model_option: ModelOption):
        self.circuit_option: T = circuit_option
        self.model_option: ModelOption = model_option

    def process_counts(self, counts: Dict) -> Tuple[List[List[int]], List[float]]:
        collapse_state = [[int(char) for char in state] for state in counts.keys()]
        total_count = sum(counts.values())
        probs = [count / total_count for count in counts.values()]
        return collapse_state, probs

    @abstractmethod
    def get_num_params(self):
        pass

    @abstractmethod
    def inference(self, params):
        pass

    @abstractmethod
    def create_circuit(self) -> QuantumCircuit:
        pass

    def get_circuit_cost_func(self):
        def circuit_cost_func(params):
            collapse_state, probs = self.inference(params)
            costs = 0
            for value, prob in zip(collapse_state, probs):
                costs += self.model_option.obj_func(value) * prob
            return costs

        return circuit_cost_func

    def draw(self) -> None:
        pass

    def analyze(self):
        pass
