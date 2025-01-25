from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable
import numpy as np

class ConstraintsDemo(LcboModel):
    def __init__(self, num_qubit: int, num_cnst: int) -> None:
        super().__init__()
        # 投资项目总数
        self.num_qubit = num_qubit
        self.num_cnst = num_cnst
        assert num_qubit % 2 == 0
        assert num_cnst <= num_qubit / 2
        x = self.addVars(num_qubit, name="x")
        self.setObjective(sum(np.random.randint(1, 21) * x[i] for i in range(num_qubit // 2)), 'max')
        self.addConstr((sum(x[i] for i in range(num_qubit))) == num_qubit / 2)
        self.addConstrs((x[2 * j] + x[2 * j + 1] == 1 for j in range(num_cnst)))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))
        for i in range(self.num_qubit):
            if i % 2 == 0:
                fsb_lst[i] = 1
        self.fill_feasible_solution(fsb_lst)
        return fsb_lst