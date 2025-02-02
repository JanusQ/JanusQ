from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable, Tuple

class CapitalBudgetingProblem(LcboModel):
    def __init__(self, num_projects, revenue: Iterable[int], budget: int, dependence: Iterable[Tuple]) -> None:
        super().__init__()
        # 投资项目总数
        self.num_projects = num_projects
        self.revenue = revenue
        self.budget = budget
        self.dependence = dependence
         
        x = self.addVars(self.num_projects, name="x")
        self.setObjective(sum(self.revenue[i] * x[i] for i in range(self.num_projects)), 'max')
        self.addConstr((sum(x[i] for i in range(self.num_projects))) <= self.budget)
        self.addConstrs((x[i] <= x[j] for i, j in dependence))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))
        self.fill_feasible_solution(fsb_lst)
        return fsb_lst

import random
import itertools

def generate_cbp(num_problems_per_scale, scale_list, min_value=1, max_value=20):
    def generate_dependency_pairs(n, m):
        # 生成所有可能的依赖对 (x1, x2)，其中 x1 != x2
        all_pairs = list(itertools.permutations(range(0, n), 2))
        
        # 随机选择 m 对
        if m > len(all_pairs):
            raise ValueError("m is too large, can't generate that many unique pairs.")
        
        selected_pairs = random.sample(all_pairs, m)
        
        return selected_pairs
    def generate_random_cpb(num_problems, idx_scale, num_projects, budget, num_dependences, min_value=1, max_value=20):
        problems = []
        configs = []
        for _ in range(num_problems):
            revenue = [random.randint(min_value, max_value) for _ in range(num_projects)]
            dependency_lst = generate_dependency_pairs(num_projects, num_dependences)
            problem = CapitalBudgetingProblem(num_projects, revenue, budget, dependency_lst)
            print("<<<<<<<<<")
            print(dependency_lst)
            print("<<<<<<<<<")
            print(problem.driver_bitstr)
            if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row) : 
                problems.append(problem)
                configs.append((idx_scale, len(problem.variables), len(problem.lin_constr_mtx), num_projects, revenue))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_project, budget, num_dependences) in enumerate(scale_list):
        problems, configs = generate_random_cpb(num_problems_per_scale, idx_scale, num_project, budget, num_dependences, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)
    
    return problem_list, config_list
