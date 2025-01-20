import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from chocoq.solvers.abstract_solver import Solver
from chocoq.solvers.optimizers import Optimizer
from chocoq.solvers.options import CircuitOption, OptimizerOption, ModelOption
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel

from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.circuit_components import obj_compnt, commute_compnt, penalty_decompose


class PenaltyCircuit(QiskitCircuit[CircuitOption]):
    def __init__(self, circuit_option: CircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.search_circuit()

    def get_num_params(self):
        return self.circuit_option.num_layers * 2
    
    def inference(self, params):
        final_qc = self.inference_circuit.assign_parameters(params)
        counts = self.circuit_option.provider.get_counts_with_time(final_qc, shots=self.circuit_option.shots)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs

    def search_circuit(self) -> QuantumCircuit:
        num_layers = self.circuit_option.num_layers
        num_qubits = self.model_option.num_qubits

        qc = QuantumCircuit(num_qubits, num_qubits)
        Ho_params = [Parameter(f'Ho_params[{i}]') for i in range(num_layers)]
        Hd_params = [Parameter(f'Hd_params[{i}]') for i in range(num_layers)]
        
        for i in range(num_qubits):
            qc.h(i)

        for layer in range(num_layers):
            obj_compnt(qc, Ho_params[layer], self.model_option.obj_dct)
            penalty_decompose(qc, self.model_option.lin_constr_mtx, Ho_params[layer], num_qubits)
            for i in range(num_qubits):
                qc.rx(Hd_params[layer], i)

        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        transpiled_qc = self.circuit_option.provider.transpile(qc)
        return transpiled_qc
    
class PenaltySolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int,
        shots: int = 1024,
    ):
        super().__init__(prb_model, optimizer)
        self.circuit_option = CircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = PenaltyCircuit(self.circuit_option, self.model_option)
        return self._circuit