from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable


class JobSchedulingProblem(LcboModel):
    def __init__(self, num_jobs: int, num_machines: int, cost: Iterable[Iterable], capacity: Iterable) -> None:
        super().__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.cost = cost
        self.capacity = capacity
        self.x = x = self.addVars(self.num_jobs, self.num_machines, name="x")
        self.setObjective(sum(self.cost[i][j] * x[i, j] for i in range(self.num_jobs) for j in range(self.num_machines)), "min")
        self.addConstrs(sum(x[i, j] for j in range(self.num_machines)) == 1 for i in range(self.num_jobs))
        self.addConstrs(sum(x[i, j] for i in range(self.num_jobs)) <= self.capacity[j] for j in range(self.num_machines))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))

        t_machine = 0
        t_load = 0
        for job_idx in range(self.num_jobs):
            # 找到一个空的机子，如果当前的承载已经超过容量了，就跳到下一个
            while t_load >= self.capacity[t_machine]:
                t_load = 0
                t_machine += 1
            fsb_lst[self.var_idx(self.x[job_idx, t_machine])] = 1
            t_load += 1


        self.fill_feasible_solution(fsb_lst)
        return fsb_lst


import random

def generate_jsp(num_problems_per_scale, scale_list, min_value=1, max_value=20):
    def generate_capacities(num_machines, total_capacity):
        # 生成 num_machines - 1 个随机分割点
        splits = sorted(random.sample(range(1, total_capacity), num_machines - 1))

        # 根据分割点计算每个机器的容量
        capacities = [splits[0]] + [splits[i] - splits[i - 1] for i in range(1, len(splits))] + [total_capacity - splits[-1]]

        return capacities

    def generate_random_jsp(num_problems, idx_scale, num_jobs, num_machines, total_capacity, min_value=1, max_value=20):
        problems = []
        configs = []
        for _ in range(num_problems):
            while True:
                cost = [[random.randint(min_value, max_value) for _ in range(num_machines)] for _ in range(num_jobs)]
                capacity = generate_capacities(num_machines, total_capacity)
                problem = JobSchedulingProblem(num_jobs, num_machines, cost, capacity)
                if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row): 
                    break
            problems.append(problem)
            configs.append((idx_scale, len(problem.variables), len(problem.lin_constr_mtx), num_jobs, num_machines, cost, capacity))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_jobs, num_machines, total_capacity) in enumerate(scale_list):
        problems, configs = generate_random_jsp(num_problems_per_scale, idx_scale, num_jobs, num_machines, total_capacity, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)
    
    return problem_list, config_list