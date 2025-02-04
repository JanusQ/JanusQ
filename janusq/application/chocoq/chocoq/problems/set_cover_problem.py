from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable, List, Tuple
class SetCoverProblem(LcboModel):
    def __init__(self, num_sets: int, num_elements, list_covering: List[List]) -> None:
        super().__init__()
        self.num_sets = num_sets
        self.num_elements = num_elements
        self.list_covering = list_covering
        self.num_elements = len(self.list_covering)

        self.x = x = self.addVars(num_sets, name='x')
        self.setObjective(sum(x[i] for i in range(num_sets)), 'min')
        self.addConstrs((sum(x[i] for i in range(num_sets) if element in list_covering[i]) >= 1) for element in range(num_elements))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))
        have_cover_set = set()
        for i in range(self.num_sets):
            fsb_lst[i] = 1
            have_cover_set.update(self.list_covering[i])
            if len(have_cover_set) == self.num_elements:
                break
        self.fill_feasible_solution(fsb_lst)
        return fsb_lst
    

# ////////////////////////////////////////////////////

import random

def generate_scp(num_problems_per_scale, scale_list) -> Tuple[List[List[SetCoverProblem]], List[List[Tuple]]]:
    def generate_random_scp_data(s, c):
        # 初始化结果列表
        scp_data = [[] for _ in range(s)]
        
        # 确保每个点至少被一个集合覆盖
        all_points = list(range(c))
        random.shuffle(all_points)
        
        # 将每个点分配给一个随机的集合，确保每个点至少被覆盖一次
        for i, point in enumerate(all_points):
            scp_data[i % s].append(point)
        
        # 随机增加 c 次覆盖关系
        total_additions = 0
        while total_additions <  s / 3.8 * c:
            # 随机选择一个集合
            chosen_set = random.randint(0, s - 1)
            # 随机选择一个点
            new_point = random.randint(0, c - 1)
            
            # 确保不重复添加已存在的点
            if new_point not in scp_data[chosen_set]:
                scp_data[chosen_set].append(new_point)
                total_additions += 1
        
        return scp_data
    
    def generate_random_scp(num_problems, idx_scale, num_sets, num_elements):
        problems = []
        configs = []
        for _ in range(num_problems):
            while True:
                covering = generate_random_scp_data(num_sets, num_elements)
                problem = SetCoverProblem(num_sets, num_elements, covering)
                if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row): 
                    break
            problems.append(problem)
            configs.append((idx_scale, len(problem.variables), len(problem.lin_constr_mtx), num_sets, num_elements, covering))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_sets, num_elements) in enumerate(scale_list):
        problems, configs = generate_random_scp(num_problems_per_scale, idx_scale, num_sets, num_elements)
        problem_list.append(problems)
        config_list.append(configs)
    
    return problem_list, config_list