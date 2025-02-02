from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable, List, Tuple
from itertools import combinations
import numpy as np
class TravelingSalesmanProblem(LcboModel):
    def __init__(self, num_cities: int, distance_matrix: List[List[int]]= None):
        super().__init__()
        
        self.num_cities = num_cities
        if self.num_cities < 2:
            raise ValueError("Number of cities must be at least 2")
        self.distance_matrix = distance_matrix
        if self.distance_matrix is None:
            self.locations = np.random.rand(num_cities, 2)
            self.distance_matrix = np.zeros((num_cities, num_cities))
            for i in range(num_cities):
                for j in range(num_cities):
                    self.distance_matrix[i][j] = int(np.linalg.norm(self.locations[i] - self.locations[j])*1e3)
           
        self.x = self.addVars(self.num_cities, self.num_cities, name='x')
        print('distance_matrix[0][1]',self.distance_matrix[0][1])
        self.setObjective(sum(self.distance_matrix[i][j] * self.x[i, j] for i in range(self.num_cities) for j in range(self.num_cities)), 'min')
        self.addConstrs(sum(self.x[i, j] for j in range(self.num_cities)) == 1 for i in range(self.num_cities))
        # for s in range(2, num_cities):  # Subset size must be at least 2
        #     for subset in combinations(range(1, num_cities), s):
        #         # model.addConstr(quicksum(x[i, j] for i in subset for j in subset if i != j) <= len(subset) - 1)
        #         self.addConstr(sum(self.x[i,j] for i in subset for j in subset if i != j) <= len(subset) - 1)
        self.addConstrs(self.x[i, i] == 0 for i in range(self.num_cities))
        self.addConstrs(sum(self.x[i, j] for i in range(self.num_cities)) == 1 for j in range(self.num_cities))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst1 = np.zeros(len(self.variables))
        for i in range(self.num_cities):
            # fsb_lst[i*self.num_cities+(i+1)%self.num_cities] = 1
            fsb_lst1[((i+1)%self.num_cities)*self.num_cities+i] = 1
        self.fill_feasible_solution(fsb_lst1)
        fsb_lst2 = np.zeros(len(self.variables))
        for i in range(self.num_cities):
            fsb_lst2[i*self.num_cities+(i+1)%self.num_cities] = 1
            # fsb_lst[((i+1)%self.num_cities)*self.num_cities+i] = 1
        self.fill_feasible_solution(fsb_lst2)
        u = fsb_lst2 - fsb_lst1
        print('u',u)
        return fsb_lst1
    


class TSPHalf(LcboModel):
    def __init__(self, num_cities: int, distance_matrix: List[List[int]]= None):
        super().__init__()
        
        self.num_cities = num_cities
        if self.num_cities < 2:
            raise ValueError("Number of cities must be at least 2")
        self.distance_matrix = distance_matrix
        if self.distance_matrix is None:
            self.locations = np.random.rand(num_cities, 2)
            self.distance_matrix = np.zeros((num_cities, num_cities))
            for i in range(num_cities):
                for j in range(num_cities):
                    self.distance_matrix[i][j] = int(np.linalg.norm(self.locations[i] - self.locations[j])*1e1)
        vars = self.addVars(int(num_cities*(num_cities-1)/2), name="edge")
        x = {}
        idx = 0
        self.idx_map = {}
        for (i,j) in combinations(range(num_cities), 2):
            x[i,j] = vars[idx]
            x[j,i] = vars[idx]
            self.idx_map[i,j] = idx
            self.idx_map[j,i] = idx
            idx += 1
            
        for i in range(num_cities):
            x[i,i] = 0

        self.setObjective(sum(self.distance_matrix[i][j] * x[i,j] for i in range(num_cities) for j in range(num_cities)), 'min')

        self.addConstrs((sum(x[j,i] for j in range(i))+ sum(x[i,j] for j in range(i+1, num_cities)) == 2) for i in range(num_cities))

        

        for s in range(2, num_cities):  # Subset size must be at least 2
            for subset in combinations(range(1, num_cities), s):
                self.addConstr(sum(x[i, j] for i in subset for j in subset if i < j) <= len(subset) - 1)


    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst1 = np.zeros(len(self.variables))
        for i in range(self.num_cities):
            fsb_lst1[self.idx_map[i, (i+1)%self.num_cities]] = 1
        self.fill_feasible_solution(fsb_lst1)
        return fsb_lst1
    


def generate_tsp(num_problems_per_scale,scale_list):
    # 给定点数和组数, 给出所有分配方案
    problem_list = []
    config_list = []
    for num_cities in scale_list:
        problem_scales = []
        for i in range(num_problems_per_scale):
            prob = TSPHalf(num_cities)
            problem_scales.append(prob)
        
        problem_list.append(problem_scales)
        config_list.append([[num_cities]*num_problems_per_scale])
        

    return problem_list, config_list