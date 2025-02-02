from dataclasses import dataclass, field
from typing import List, Callable, Tuple

@dataclass(kw_only=True)
class OptimizerOption:
    obj_dir: int = None
    num_params: int = None
    cost_func: Callable = None

    # params_optimization_method: str = 'COBYLA'
    # max_iter: int = 30
    # opt_id: any = None
    # use_local_params: bool = False
    # #
    # circuit_cost_function: Callable = None

@dataclass(kw_only=True)
class CobylaOptimizerOption(OptimizerOption):
    max_iter: int = 50

@dataclass(kw_only=True)
class AdamOptimizerOption(OptimizerOption):
    max_iter: int = 50
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999


