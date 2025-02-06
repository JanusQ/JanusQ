import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ...solvers.abstract_solver import Solver
from ...solvers.optimizers import Optimizer
from ...solvers.options import CircuitOption, OptimizerOption, ModelOption
from ...solvers.options.circuit_option import ChCircuitOption
from ...model import LinearConstrainedBinaryOptimization as LcboModel
from ...utils import iprint
from ...utils.linear_system import to_row_echelon_form, greedy_simplification_of_transition_Hamiltonian
from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.circuit_components import obj_compnt, new_compnt
from .explore.qto_search import QtoSearchSolver

class QtoSimplifyDiscardCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        # iprint(self.model_option.feasible_state)
        # iprint(self.model_option.Hd_bitstr_list)
        # exit()
        self.inference_circuit = self.create_circuit()

    def get_num_params(self):
        return len(self.model_option.Hd_bitstr_list)
    
    def inference(self, params):
        final_qc = self.inference_circuit.assign_parameters(params)
        counts = self.circuit_option.provider.get_counts_with_time(final_qc, shots=self.circuit_option.shots)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs

    def create_circuit(self) -> QuantumCircuit:
        
        mcx_mode = self.circuit_option.mcx_mode
        num_layers = self.circuit_option.num_layers
        num_qubits = self.model_option.num_qubits
        
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))

        qc = self.circuit_option.provider.transpile(qc)
        
        num_bitstrs = len(self.model_option.Hd_bitstr_list)
        Hd_params_lst = [Parameter(f"Hd_params[{i}, {j}]") for j in range(num_bitstrs) for i in range(num_layers)]

        for i in np.nonzero(self.model_option.feasible_state)[0]:
            qc.x(i)

        new_compnt(
            qc,
            Hd_params_lst,
            self.model_option.Hd_bitstr_list,
            anc_idx,
            mcx_mode,
        )

        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        transpiled_qc = self.circuit_option.provider.transpile(qc)
        return transpiled_qc

class QtoSimplifyDiscardSolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int = 1,
        shots: int = 1024,
        mcx_mode: str = "constant",
    ):
        super().__init__(prb_model, optimizer)
        # 根据排列理论，直接赋值
        num_layers = len(self.model_option.Hd_bitstr_list)
        # 贪心减少非零元 优化跃迁哈密顿量
        self.model_option.Hd_bitstr_list = greedy_simplification_of_transition_Hamiltonian(self.model_option.Hd_bitstr_list)
        
        self.circuit_option = ChCircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode,
        )
        search_solver = QtoSearchSolver(
            prb_model=prb_model,
            optimizer=optimizer,
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode
        )

        _, set_basis_lists, _ = search_solver.search()

        min_id = 0
        max_id = 0

        useful_idx = []
        already_set = set()
        if len(set_basis_lists[0]) != 1:
            useful_idx.append(0)

        already_set.update(set_basis_lists[0])

        for i in range(1, len(set_basis_lists)):
            # if len(set_basis_lists[i - 1]) == 1 and min_id == i - 1:
            #     min_id = i
            if set_basis_lists[i] - already_set:
                already_set.update(set_basis_lists[i])
                max_id = i
        iprint(f'range({min_id}, {max_id})')
        Hd_bitstr_list = np.tile(self.model_option.Hd_bitstr_list, (num_layers, 1))
        self.model_option.Hd_bitstr_list = [item for i, item in enumerate(Hd_bitstr_list) if i >= min_id and i < max_id]


    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = QtoSimplifyDiscardCircuit(self.circuit_option, self.model_option)
        return self._circuit


