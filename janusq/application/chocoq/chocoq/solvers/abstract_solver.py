from abc import ABC, abstractmethod

from chocoq.solvers.optimizers import Optimizer
from chocoq.utils import iprint
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel

from .options import CircuitOption
from .options.model_option import ModelOption
from .qiskit.circuit import QiskitCircuit
from .data_analyzer import DataAnalyzer
import time

class Solver(ABC):
    def __init__(self, prb_model: LcboModel, optimizer: Optimizer):
        if isinstance(prb_model, LcboModel):
            self.model_option = prb_model.to_model_option()
        elif isinstance(prb_model, ModelOption):
            self.model_option = prb_model
        else:
            raise TypeError(f"Expected LcboModel or ModelOption, got {type(prb_model)}")
        self.optimizer: Optimizer = optimizer
        self.circuit_option: CircuitOption = None

        self._circuit = None

        self.collapse_state_lst = None
        self.probs_lst = None
        self.iter_count = None
        self.evaluation_lst = None

        self.solver_start_time = time.perf_counter()  # 记录开始时间用于计算端到端时间

    @property
    @abstractmethod
    def circuit(self) -> QiskitCircuit:
        pass

    def solve(self):
        self.optimizer.optimizer_option.obj_dir = self.model_option.obj_dir
        self.optimizer.optimizer_option.cost_func = self.circuit.get_circuit_cost_func()
        self.optimizer.optimizer_option.num_params = self.circuit.get_num_params()
        best_params, self.iter_count = self.optimizer.minimize()
        iprint(best_params)
        self.collapse_state_lst, self.probs_lst = self.circuit.inference(best_params)
        solver_end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
        self.end_to_end_time = solver_end_time - self.solver_start_time
        return self.collapse_state_lst, self.probs_lst, self.iter_count
    
    # def solve(self):
    #     result = self.solve()
    #     end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
        # self.end_to_end_time = end_time - self.start_time  # 计算耗时
    #     return result
    
    def evaluation(self):
        """在调用过solve之后使用"""
        assert self.collapse_state_lst is not None

        model_option = self.model_option
        data_analyzer = DataAnalyzer(
            collapse_state_lst = self.collapse_state_lst, 
            probs_lst = self.probs_lst, 
            obj_func = model_option.obj_func, 
            best_cost = model_option.best_cost,
            lin_constr_mtx = model_option.lin_constr_mtx
        )
        data_metrics_lst = data_analyzer.summary()
        # 把 iteration_count 加到 指标 结尾，构成完整评估
        self.evaluation_lst = data_metrics_lst + [self.iter_count]
        return self.evaluation_lst
        
    def circuit_analyze(self, metrics_lst):
        return self.circuit.analyze(metrics_lst)
    
    def time_analyze(self):
        quantum = self.circuit_option.provider.quantum_circuit_execution_time
        classcial = self.end_to_end_time - quantum
        return classcial, quantum
    
    def run_counts(self):
        return self.circuit_option.provider.run_count

    # def __hash__(self):
    #     # 使用一个元组的哈希值作为对象的哈希值
    #     return hash(self.name)

    # def __eq__(self, other):
    #     if isinstance(other, Solver):
    #         return self.name == other.name
    #     return False
