import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from chocoq.solvers.abstract_solver import Solver
from chocoq.solvers.optimizers import Optimizer
from chocoq.solvers.options import CircuitOption, OptimizerOption, ModelOption
from chocoq.solvers.options.circuit_option import ChCircuitOption
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from chocoq.utils import iprint
from chocoq.utils.linear_system import to_row_echelon_form, greedy_simplification_of_transition_Hamiltonian
from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.circuit_components import obj_compnt, new_compnt
from .explore.qto_search import QtoSearchSolver

class QtoSimplifyDiscardSegmentedFilterCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption, hlist: list[QuantumCircuit]):
        super().__init__(circuit_option, model_option)
        # iprint(self.model_option.feasible_state)
        # iprint(self.model_option.Hd_bitstr_list)
        # exit()
        self.inference_circuit = self.create_circuit()
        self.hlist = hlist

    def get_num_params(self):
        return len(self.hlist)
    
    def inference(self, params):
        counts = self.segmented_excute_circuit(params)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs
    
    # @property
    # def inference_circuit(self):   
    #     raise Exception("This circuit is not yet suitable for analysis")
    
    def segmented_excute_circuit(self, params) -> QuantumCircuit:
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        # self.qc = self.circuit_option.provider.transpile(qc)

        def run_and_pick(dict:dict, hdi_qc: QuantumCircuit, param):
            # iprint("--------------")
            # iprint(f'input dict: {dict}')
            dicts = []
            total_count = sum(dict.values())
            for key, value in dict.items():
                if mcx_mode == "constant":
                    qc_temp = QuantumCircuit(num_qubits + 2, num_qubits)
                elif mcx_mode == "linear":
                    qc_temp = QuantumCircuit(2 * num_qubits, num_qubits)

                for idx, key_i in enumerate(key):
                    if key_i == '1':
                        qc_temp.x(idx)
                qc_add = hdi_qc.assign_parameters([param])
                qc_temp.compose(qc_add, inplace=True)
                qc_temp.measure(range(num_qubits), range(num_qubits)[::-1])
                # iprint(f'hdi depth: {qc_temp.depth()}')
                qc_temp = self.circuit_option.provider.transpile(qc_temp)
                
                count = self.circuit_option.provider.get_counts_with_time(qc_temp, shots=self.circuit_option.shots * value // total_count)
                # origin = self.circuit_option.shots * value // total_count
                # count = self.circuit_option.provider.get_counts_with_time(qc_temp, shots=1024)
                # count = {k: round(v / 1024 * origin, 0) for k, v in count.items() if round(v / 1024 * origin, 0) > 0}

                dicts.append(count)
            # iprint(f'this hdi depth: {qc_temp.depth()}')

            # iprint(f'evolve: {dicts}')
            merged_dict = {}
            for d in dicts:
                for key, value in d.items():
                    if all([np.dot([int(char) for char in key], constr[:-1]) == constr[-1] for constr in self.model_option.lin_constr_mtx]):
                        merged_dict[key] = merged_dict.get(key, 0) + value
            # iprint(f'feasible counts: {merged_dict}')
            return merged_dict


        register_counts = {''.join(map(str, self.model_option.feasible_state)): 1}
        for i, h_tau in enumerate(self.hlist):
            register_counts = run_and_pick(register_counts, h_tau, params[i])

        return register_counts

class QtoSimplifyDiscardSegmentedFilterSolver(Solver):
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
        # self.hlist = search_solver.hlist[:1]

        hlist = search_solver.hlist
        _, set_basis_lists, _ = search_solver.search()

        min_id = 0
        max_id = 0

        useful_idx = []
        already_set = set()
        if len(set_basis_lists[0]) != 1:
            useful_idx.append(0)

        already_set.update(set_basis_lists[0])

        for i in range(1, len(set_basis_lists)):
            if len(set_basis_lists[i - 1]) == 1 and min_id == i - 1:
                min_id = i
            if set_basis_lists[i] - already_set:
                already_set.update(set_basis_lists[i])
                max_id = i
        iprint(f'range({min_id}, {max_id})')
        self.hlist = []
        hlist_len = len(hlist)
        for i in range(min_id, max_id):
            self.hlist.append(hlist[i % hlist_len])


    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = QtoSimplifyDiscardSegmentedFilterCircuit(self.circuit_option, self.model_option, self.hlist)
        return self._circuit


