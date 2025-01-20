from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable, List, Tuple
class KPartitionProblem(LcboModel):
    def __init__(self, num_points: int, block_allot: List[int], pairs_connected: List[Tuple[Tuple[int, int], int]]) -> None:
        super().__init__()
        self.num_points = num_points
        self.num_block = len(block_allot)
        self.block_allot = block_allot
        self.pairs_connected = pairs_connected
        self.num_pairs = len(pairs_connected)

        self.x = x = self.addVars(self.num_points, self.num_block, name='x')
        self.setObjective(sum(cost * (1 - sum(x[i, k] * x[j, k] for k in range(self.num_block))) for (i, j), cost in pairs_connected), 'min')
        self.addConstrs(sum(x[i, k] for k in range(self.num_block)) == 1 for i in range(num_points))
        self.addConstrs((sum(x[i, k] for i in range(num_points)) == block_allot[k]) for k in range(self.num_block))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))

        t = 0
        for b, p in enumerate(self.block_allot):
            for _ in range(p):
                fsb_lst[self.var_idx(self.x[t, b])] = 1
                t += 1

        self.fill_feasible_solution(fsb_lst)
        return fsb_lst
    

import random

def generate_kpp(num_problems_per_scale, scale_list, min_value=1, max_value=20):
    # 给定点数和组数, 给出所有分配方案
    def partition(number, k):
        answer = []
        def helper(n, k, start, current):
            if k == 1:
                if n >= start:
                    answer.append(current + [n])
                return
            for i in range(start, n - k + 2):
                helper(n - i, k - 1, i, current + [i])
        helper(number, k, 1, [])
        return answer
    def generate_random_kpp(num_problems, idx_scale, num_points, num_allot, num_pairs, min_value=1, max_value=20):
        problems = []
        configs = []
        block_allot_list = partition(num_points, num_allot)
        allot_idx = 0
        len_block_allot = len(block_allot_list)
        for _ in range(num_problems):
            while True:
                pairs_connected = set()
                while len(pairs_connected) < num_pairs:
                    u = random.randint(0, num_points - 1)
                    v = random.randint(0, num_points - 1)
                    if u != v:
                        # Ensure unique edges based on vertices only
                        edge = tuple(sorted((u, v)))
                        if edge not in pairs_connected:
                            pairs_connected.add(edge)
                pairs_connected = [((u, v), random.randint(min_value, max_value)) for (u, v) in pairs_connected]
                block_allot = block_allot_list[allot_idx]
                allot_idx = (allot_idx + 1) % len_block_allot
                problem = KPartitionProblem(num_points, block_allot, pairs_connected)
                if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row): 
                    break
            problems.append(problem)
            configs.append((idx_scale, len(problem.variables), num_points, block_allot, len(pairs_connected), pairs_connected))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_points, num_allot, num_pairs) in enumerate(scale_list):
        problems, configs = generate_random_kpp(num_problems_per_scale, idx_scale, num_points, num_allot, num_pairs, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)

    return problem_list, config_list