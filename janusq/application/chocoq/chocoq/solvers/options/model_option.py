from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Dict

@dataclass
class ModelOption:
    num_qubits: int = None
    penalty_lambda: float = None
    feasible_state: List[int] = None # field(default_factory=list)
    obj_dct: Dict[int, List] = None # field(default_factory=dict)
    lin_constr_mtx: List[List[float]] = field(default_factory=list)
    Hd_bitstr_list: List[List[int]] = field(default_factory=list)
    obj_dir: int = None
    obj_func: Callable = None
    best_cost: float = None