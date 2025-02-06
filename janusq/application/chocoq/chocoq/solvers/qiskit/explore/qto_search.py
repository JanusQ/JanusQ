import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from chocoq.solvers.abstract_solver import Solver
from chocoq.solvers.optimizers import Optimizer
from chocoq.solvers.options import CircuitOption, OptimizerOption, ModelOption
from chocoq.solvers.options.circuit_option import ChCircuitOption
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from chocoq.utils.linear_system import to_row_echelon_form, greedy_simplification_of_transition_Hamiltonian
from chocoq.utils.gadget import iprint
from ..circuit import QiskitCircuit
from ..provider import Provider
from ..circuit.circuit_components import obj_compnt, search_evolution_space_low_cost
from ..circuit.hdi_decompose import driver_component


class QtoSearchCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        iprint(self.model_option.Hd_bitstr_list)
        self.transpiled_hlist = self.transpile_hlist()
        self._hlist = self.hlist()
        self.result = self.search_circuit()

    def get_num_params(self):
        return self.circuit_option.num_layers * 2
    
    def inference(self, params):
        print("use func: search")
        exit()

    def transpile_hlist(self):
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))
        self.qc = qc
        

        transpiled_hlist = []
        for hdi_vct in self.model_option.Hd_bitstr_list:
            qc_temp: QuantumCircuit = qc.copy()
            nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
            hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
            H_param = Parameter("1")
            driver_component(qc_temp, nonzero_indices, anc_idx, hdi_bitstr, H_param, mcx_mode)
            transpiled_qc = self.circuit_option.provider.transpile(qc_temp)
            transpiled_hlist.append(transpiled_qc)
            
        return transpiled_hlist

    def hlist(self):
        """没有经过编译的逻辑电路

        Returns:
            list: 逻辑电路列表
        """
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))
        self.qc = qc
        

        hlist = []
        for hdi_vct in self.model_option.Hd_bitstr_list:
            qc_temp: QuantumCircuit = qc.copy()
            nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
            hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
            H_param = Parameter("1")
            driver_component(qc_temp, nonzero_indices, anc_idx, hdi_bitstr, H_param, mcx_mode)
            hlist.append(qc_temp)
            
        return hlist
    
    def search_circuit(self) -> QuantumCircuit:
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
        # Ho_params = np.random.rand(num_layers)
        
        # Hd_params = np.random.rand(num_layers)

        for i in np.nonzero(self.model_option.feasible_state)[0]:
            qc.x(i)
        num_basis_lists = []
        set_basis_lists = []
        depth_lists = []
        already_set = set()
        for layer in range(num_layers):
            Hd_params = np.full(num_layers, np.random.uniform(0.1, np.pi / 4 - 0.1))
            iprint(f"===== times of repetition: {layer + 1} ======")
            num_basis_list, set_basis_list, depth_list = search_evolution_space_low_cost(
                qc,
                Hd_params,
                self.model_option.Hd_bitstr_list,
                anc_idx,
                mcx_mode,
                num_qubits,
                self.circuit_option.shots * 10,
                self.circuit_option.provider,
            )
            num_basis_lists.extend(num_basis_list)
            set_basis_lists.extend(set_basis_list)
            depth_lists.extend(depth_list)
            this_time = set.union(*set_basis_list)
            iprint(num_basis_list)
            # 早停
            if this_time - already_set:
                already_set.update(this_time)
            else:
                break
        return num_basis_lists, set_basis_lists, depth_lists


class QtoSearchSolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int,
        shots: int = 1024,
        mcx_mode: str = "constant",
    ):
        super().__init__(prb_model, optimizer)
        # m 层
        num_layers = len(self.model_option.Hd_bitstr_list)

        # need to refix
        from chocoq.solvers.qiskit import DdsimProvider

        self.circuit_option = ChCircuitOption(
            provider=DdsimProvider(),
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = QtoSearchCircuit(self.circuit_option, self.model_option)
        return self._circuit

    def search(self):
        return self.circuit.result
    
    @property
    def transpiled_hlist(self):
        return self.circuit.transpiled_hlist
    
    @property
    def hlist(self):
        return self.circuit._hlist
