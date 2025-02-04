from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable, List, Tuple

class FacilityLocationProblem(LcboModel):
    def __init__(self, num_demands: int, num_facilities: int, cost_service: Iterable[Iterable], cost_facilities: Iterable) -> None:
        super().__init__()
        self.num_demands = num_demands
        self.num_facilities = num_facilities
        self.cost_service = cost_service 
        self.cost_facilities = cost_facilities
         
        x = self.addVars(self.num_facilities, name="x")
        y = self.addVars(self.num_demands, self.num_facilities, name="y")
        self.setObjective(sum(self.cost_service[i][j] * y[i, j] for i in range(self.num_demands) for j in range(self.num_facilities)) + sum(self.cost_facilities[j] * x[j] for j in range(self.num_facilities)), 'min')
        self.addConstrs(sum(y[i, j] for j in range(self.num_facilities)) == 1 for i in range(self.num_demands))
        self.addConstrs(y[i, j] <= x[j] for i in range(self.num_demands) for j in range(self.num_facilities))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))
        fsb_lst[0] = 1
        for i in range(self.num_demands):
            fsb_lst[self.num_facilities + self.num_facilities * i] = 1

        self.fill_feasible_solution(fsb_lst)
        return fsb_lst

import random

def generate_flp(num_problems_per_scale, scale_list, min_value=1, max_value=20) -> Tuple[List[List[FacilityLocationProblem]], List[List[Tuple]]]:
    def generate_random_flp(num_problems, idx_scale, num_demands, num_facilities, min_value=1, max_value=20):
        problems = []
        configs = []
        for _ in range(num_problems):
            while True:
                transport_costs = [[random.randint(min_value, max_value) for _ in range(num_facilities)] for _ in range(num_demands)]
                facility_costs = [random.randint(min_value, max_value) for _ in range(num_facilities)]
                problem = FacilityLocationProblem(num_demands, num_facilities, transport_costs, facility_costs)
                if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row): 
                    break
            problems.append(problem)
            configs.append((idx_scale, len(problem.variables), num_demands, num_facilities, transport_costs, facility_costs))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_demands, num_facilities) in enumerate(scale_list):
        problems, configs = generate_random_flp(num_problems_per_scale, idx_scale, num_demands, num_facilities, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)
    
    return problem_list, config_list