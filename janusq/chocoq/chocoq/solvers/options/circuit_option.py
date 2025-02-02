from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Dict, Union

from .model_option import ModelOption
from ..qiskit.provider import Provider
# from qiskit.providers import Backend, BackendV2
# from qiskit.transpiler import PassManager


@dataclass(kw_only=True)
class CircuitOption():
    provider: Provider
    num_layers: int
    shots: int = 1024

    # num_qubits: int = None
    # penalty_lambda: float = None
    # feasible_state: List[int] = None # field(default_factory=list)
    # obj_dct: Dict[int, List] = None # field(default_factory=dict)
    # lin_constr_mtx: List[List[float]] = field(default_factory=list)
    # Hd_bitstr_list: List[List[int]] = field(default_factory=list)
    # obj_func: Callable = None
    
    
    # need_draw: bool = False
    # use_decompose: bool = False
    # use_serialization : bool = False # 不分解情况的可选项
    # circuit_type: str = 'qiskit'
    # mcx_mode: str = 'constant'  # 'constant' for 2 additional ancillas with linear depth, 'linear' for n-1 additional ancillas with logarithmic depth
    # backend: str = 'FakeAlmadenV2' #'FakeQuebec' # 'AerSimulator'\
    # feedback: List = field(default_factory=list)
    # use_IBM_service_mode: str = None
    # use_free_IBM_service: bool = True
    # use_fake_IBM_service: bool = False
    # # cloud_manager: CloudManager = None
    # # 
    # # log_depth: bool = False
    # num_qubits: int = 0
    # algorithm_optimization_method: str = 'commute'
    # objective_func: Callable = None
    # objective_func_term_list: List[List[Tuple[List[int], float]]] = field(default_factory=list)
    # constraints_for_cyclic: List[List[float]] = field(default_factory=list)
    # constraints_for_others: List[List[float]] = field(default_factory=list)

@dataclass(kw_only=True)
class ChCircuitOption(CircuitOption):
    mcx_mode: str # 'constant' for 2 additional ancillas with linear depth, 'linear' for n - 1 additional ancillas with logarithmic depth

