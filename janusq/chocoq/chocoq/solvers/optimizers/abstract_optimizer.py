from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar, Generic

from chocoq.solvers.options import OptimizerOption


# np.random.seed(0x7f)

class Optimizer(ABC):
    def __init__(self):
        self.optimizer_option: OptimizerOption = None

    def obj_dir_trans(self, obj_dir, obj_func):
        """如果是max问题 转换成min问题"""
        def trans_obj_func(*args, **kwargs):
            return obj_dir * obj_func(*args, **kwargs)
        return trans_obj_func

    @abstractmethod
    def minimize(self):
        pass

    def _initialize_params(self, num_params):
        """初始化电路参数"""
        # return 0 * np.ones(num_params)
        # return np.pi / 4 * np.ones(num_params)
        return 2 * np.pi * np.random.uniform(0, 1, num_params)