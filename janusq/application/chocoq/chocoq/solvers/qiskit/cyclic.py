import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from chocoq.solvers.abstract_solver import Solver
from chocoq.solvers.optimizers import Optimizer
from chocoq.solvers.options import CircuitOption, OptimizerOption, ModelOption
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel

from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.circuit_components import obj_compnt, cyclic_compnt, penalty_decompose


class CyclicCircuit(QiskitCircuit[CircuitOption]):
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
        model_option = self.model_option
        constr_cyclic = []
        constr_others = []

        qc = QuantumCircuit(num_qubits, num_qubits)
        Ho_params = [Parameter(f'Ho_params[{i}]') for i in range(num_layers)]
        Hd_params = [Parameter(f'Hd_params[{i}]') for i in range(num_layers)]

        for constr in model_option.lin_constr_mtx:
            if set(constr[:-1]).issubset({0, 1}) or set(constr[:-1]).issubset({0, -1}):
                constr_cyclic.append(constr)
            else:
                constr_others.append(constr)

        qubits_cyclic = {item for sublist in constr_cyclic for item in np.nonzero(sublist[:-1])[0]}
        qubits_others = set(range(num_qubits)) - qubits_cyclic
        
        for i in set(np.nonzero(self.model_option.feasible_state)[0]).intersection(qubits_cyclic):
            qc.x(i)
        for i in qubits_others:
            qc.h(i)

        for layer in range(num_layers):
            obj_compnt(qc, Ho_params[layer], self.model_option.obj_dct)
            penalty_decompose(qc, constr_others, Ho_params[layer], num_qubits)
            cyclic_compnt(
                qc,
                Hd_params[layer],
                constr_cyclic,
            )
            for i in qubits_others:
                qc.rx(Hd_params[layer], i)

        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        transpiled_qc = self.circuit_option.provider.transpile(qc)
        return transpiled_qc


class CyclicSolver(Solver):
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
            self._circuit = CyclicCircuit(self.circuit_option, self.model_option)
        return self._circuit
