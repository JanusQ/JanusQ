import itertools
import random
from quBLP.utils.gadget import iprint
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.problemtemplate import One_Hdi as OH

random.seed(0x7f)

def generate_flp(num_problems_per_scale, scale_list, min_value=1, max_value=20):
    def generate_random_flp(num_problems, idx_scale, m, n, min_value=1, max_value=20):
        problems = []
        configs = []
        for _ in range(num_problems):
            transport_costs = [[random.randint(min_value, max_value) for _ in range(n)] for _ in range(m)]
            facility_costs = [random.randint(min_value, max_value) for _ in range(n)]
            problem = FLP(m, n, transport_costs, facility_costs)
            if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row) : 
                problems.append(problem)
                configs.append((idx_scale, problem.num_variables, m, n, transport_costs, facility_costs))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (m, n) in enumerate(scale_list):
        problems, configs = generate_random_flp(num_problems_per_scale, idx_scale, m, n, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)
    
    return problem_list, config_list

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
            cost_color = [random.randint(min_value, max_value) for _ in range(num_nodes)]
            problem = GCP(num_nodes, edges_comb, cost_color)
            if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row) : 
                problems.append(problem)
                configs.append((idx_scale, problem.num_variables, num_nodes, num_edges, edges_comb, cost_color))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_nodes, num_edges) in enumerate(scale_list):
        problems, configs = generate_all_gcp(idx_scale, num_nodes, num_edges, max_problems_per_scale, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)

    return problem_list, config_list

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
            problem = KPP(num_points, block_allot, pairs_connected)
            if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row) : 
                problems.append(problem)
                configs.append((idx_scale, problem.num_variables, num_points, block_allot, len(pairs_connected), pairs_connected))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_points, num_allot, num_pairs) in enumerate(scale_list):
        problems, configs = generate_random_kpp(num_problems_per_scale, idx_scale, num_points, num_allot, num_pairs, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)

    return problem_list, config_list

def generate_oh(n_start, n_end):
    problem_list = []
    config_list = []
    for idx_scale, n in enumerate(range(n_start, n_end)):
        problem = OH(n)
        problem_list.append(problem)
        config_list.append((idx_scale, n))
    return problem_list, config_list


if __name__ == '__main__':
    # flp_problems_pkg, flp_configs_pkg = generate_flp(10, [(1, 2), (2, 3), (3, 3), (3, 4)], 1, 20)
    # gcp_problems_pkg, gcp_configs_pkg = generate_gcp(10, [(3, 2), (3, 2), (4, 2), (4, 3)])
    # kpp_problems_pkg, kpp_configs_pkg = generate_kpp(10, [(4, 2, 3), (6, 3, 5), (8, 3, 7), (9, 3, 8)], 1, 20)

    # # problems = flp_problems + gcp_problems + kpp_problems
    # # tem = [p for prb in problems for p in prb]
    # # print(tem)
    # # exit()
    # all_configs = [flp_configs_pkg, gcp_configs_pkg, kpp_configs_pkg]
    # problem_types = ["FLP", "GCP", "KPP"]
    # for problem_type, configs in zip(problem_types, all_configs):
    #     print(f"{problem_type}:")
    #     print(sum(len(row) for row in configs))
    
    # configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg
    # for pkid, configs in enumerate(configs_pkg):
    #     for pbid, problem in enumerate(configs):
    #         print(f'{pkid}-{pbid}: {problem}')

    a, b =  generate_oh(5, 13)
    for c, d in zip(a, b):
        print (c, d)
    