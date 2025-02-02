import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from chocoq.solvers.abstract_solver import Solver
from chocoq.solvers.optimizers import Optimizer
from chocoq.solvers.options import CircuitOption, OptimizerOption, ModelOption
from chocoq.solvers.options.circuit_option import ChCircuitOption
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel

import pennylane as qml
from chocoq.utils.gadget import iprint
from .circuit.pennylane_decompose import driver_component as driver_component_pennylane
from .circuit.build_circuit import PennylaneCircuit
class ChocoCircuit(PennylaneCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.create_circuit()
        print(self.model_option.Hd_bitstr_list)
        exit()

    def get_num_params(self):
        return self.circuit_option.num_layers* len(self.model_option.Hd_bitstr_list)
    
    def inference(self, params):
        qml_probs = self.inference_circuit(params)
        # print("qml_probs",qml_probs)
        bitstrsindex = np.nonzero(qml_probs)[0]
        probs = qml_probs[bitstrsindex]
        collapse_state = [[int(j) for j in list(bin(i)[2:].zfill(self.model_option.num_qubits))] for i in bitstrsindex]
        return collapse_state, probs

    def create_circuit(self):
        num_layers = self.circuit_option.num_layers
        num_qubits = self.model_option.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits + 1)
        Hd_bits_list = self.model_option.Hd_bitstr_list
        
        @qml.qnode(dev)
        def circuit_commute(params):
            for i in np.nonzero(self.model_option.feasible_state)[0]:
                qml.PauliX(i)
            for layer in range(num_layers):
                # 惩罚约束
                j =0 
                for bit_strings in range(len(Hd_bits_list)):
                    hd_bits = Hd_bits_list[bit_strings]
                    nonzero_indices = np.nonzero(hd_bits)[0]
                    hdi_string = [0 if x == -1 else 1 for x in hd_bits if x != 0]
                    driver_component_pennylane(nonzero_indices, [num_qubits] ,hdi_string, params[layer * len(Hd_bits_list) + j])
                    j += 1
            return qml.probs(wires=range(num_qubits))
            # return qml.probs(wires=[0,1])
            # return qml.expval(self.model_option.pauli_terms)
        # from jax import numpy as jnp
        # weights = jnp.array([0.1] * (num_layers * len(Hd_bits_list)))
        # gradient = qml.gradients.param_shift(circuit_commute)(weights)
        # print("gradient",gradient)
        return circuit_commute

class ChocoCleverSolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        num_layers: int,
        shots: int = 1024,
        mcx_mode: str = "constant",
    ):
        super().__init__(prb_model, optimizer)
        self.circuit_option = ChCircuitOption(
            num_layers=num_layers,
            shots=shots,
            provider=None,
            mcx_mode=mcx_mode,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = ChocoCircuit(self.circuit_option, self.model_option)
        return self._circuit


