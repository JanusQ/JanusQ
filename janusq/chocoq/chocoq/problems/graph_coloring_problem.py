from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel, fast_mul
from typing import Iterable, List, Tuple
from chocoq.utils import iprint

class GraphColoringProblem(LcboModel):
    def __init__(self, num_graphs: int, pairs_adjacent: List[Tuple[int, int]], cost_color: List[int]) -> None:
        super().__init__()
        self.num_graphs = num_graphs
        self.pairs_adjacent = pairs_adjacent
        self.num_adjacent = len(pairs_adjacent)
        self.cost_color = cost_color
        # 最坏情况每个图一个颜色
        self.num_colors = num_graphs
        self.x = x = self.addVars(self.num_graphs, self.num_colors, name="x")
        self.setObjective(sum(cost_color[k] *  (1 - fast_mul(1 - x[i, k] for i in range(num_graphs))) for k in range(self.num_colors)) , 'min')
        self.addConstrs(sum(x[i, k] for k in range(self.num_colors)) == 1 for i in range(num_graphs))
        self.addConstrs(x[i, k] + x[j, k] <= 1 for i, j in pairs_adjacent for k in range(self.num_colors))


    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))

        for i in range(self.num_graphs):
            fsb_lst[self.var_idx(self.x[i, i])] = 1

        self.fill_feasible_solution(fsb_lst)
        # print(fsb_lst)
        return fsb_lst
    
    def to_gurobi_model(self):
        # gurobi不支持高次，单独写
        import gurobipy as gp
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        
        X = model.addVars(self.num_graphs, self.num_colors, vtype=gp.GRB.BINARY, name="X")
        Y = model.addVars(self.num_colors, vtype=gp.GRB.BINARY, name="Y")
        Z = model.addVars(self.num_adjacent, self.num_colors, vtype=gp.GRB.BINARY, name="Z")
        
        model.setObjective(gp.quicksum(Y[c] * self.cost_color[c] for c in range(self.num_colors)), gp.GRB.MINIMIZE)
        model.addConstrs((gp.quicksum(X[v, i] for i in range(self.num_colors)) == 1 for v in range(self.num_graphs)), "OneColorPerGraph")
        
        for i in range(self.num_colors):
            for k, (u, w) in enumerate(self.pairs_adjacent):
                model.addConstr(Z[k, i] >= X[u, i] + X[w, i])
            for v in range(self.num_graphs):
                model.addConstr(X[v, i] <= Y[i])
        
        iprint("out feasible case is error, Orz, set 7f free")
        return model

import random
import itertools

def generate_gcp(max_problems_per_scale, scale_list, min_value=1, max_value=20):
    def generate_all_gcp(idx_scale, num_nodes, num_edges, max_problems, min_value, max_value):
        problems = []
        configs = []
        all_edges = list(itertools.combinations(range(num_nodes), 2))
        all_combinations = list(itertools.combinations(all_edges, num_edges))
        
        # 重复并截断到max_problems长度
        times_to_repeat = (max_problems // len(all_combinations)) + 1
        selected_combinations = (all_combinations * times_to_repeat)[:max_problems]
            
        for edges_comb in selected_combinations:
            while True:
                cost_color = [random.randint(min_value, max_value) for _ in range(num_nodes)]
                problem = GraphColoringProblem(num_nodes, edges_comb, cost_color)
                if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row): 
                    break
            problems.append(problem)
            configs.append((idx_scale, len(problem.variables), len(problem.lin_constr_mtx), num_nodes, num_edges, edges_comb, cost_color))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_nodes, num_edges) in enumerate(scale_list):
        problems, configs = generate_all_gcp(idx_scale, num_nodes, num_edges, max_problems_per_scale, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)

    return problem_list, config_list
