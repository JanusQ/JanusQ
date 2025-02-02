import dwave_networkx as dnx
import os
import random

from read_cnf import readCNF_intr
#DWAVE 200Q: 16 x 16

# chimera的行列数
WIDTH = 16 #8 #

NODE_ROW = WIDTH * 4
NODE_COLUMN = WIDTH * 4

NODE_CENTER_COLUMN = int(NODE_COLUMN/2)



chimera_coordinates = dnx.chimera_coordinates(WIDTH, WIDTH, 4)
linear_to_chimera = chimera_coordinates.linear_to_chimera  # chimera: 从左往右第几个，从上到下第几个，行(1)还是列(0)，（行:从左往右第几个，列:从上往下第几个）
chimera_to_linear = chimera_coordinates.chimera_to_linear


# cnf_dir_path = '/Users/siwei/workspace/CNF/UF100-430/'
# # '/Users/siwei/workspace/CNF/mix/'
# # '/Users/siwei/workspace/CNF/UUF50-218-1000/'
# # '/Users/siwei/workspace/CNF/UF100-430/'


# inter_cnf_dir_sat_path = '/Users/siwei/workspace/SAT/INTER_CNF/satisfy/128'
# # '/home/qc1/dwave-embedding/CNF/successClauses128/'
# # # 
# # #'/Users/siwei/workspace/SAT/INTER_CNF/satisfy/128'
# inter_cnf_dir_unsat_path = '/Users/siwei/workspace/SAT/INTER_CNF/unsatisfy/128'
# # '/Users/siwei/workspace/SAT/INTER_CNF/satisfy/128'
# # '/home/qc1/dwave-embedding/CNF/failClauses128/'
# # #'/Users/siwei/workspace/SAT/INTER_CNF/unsatisfy/128'
# # # 

# # cnf_path = '/Users/siwei/workspace/CNF/UF50-218-1000/uf50-02.cnf'


# sat_cnf_pathes = [
#     os.path.join(inter_cnf_dir_sat_path, f)
#     for f in os.listdir(inter_cnf_dir_sat_path)
# ]
# # random.shuffle(sat_cnf_pathes)

# unsat_cnf_pathes = []
# for f in os.listdir(inter_cnf_dir_unsat_path):
#     f = os.path.join(inter_cnf_dir_unsat_path, f)
#     # clauses, literals = readCNF_intr(f)
#     # if len(clauses) < 30 or len(clauses) > 300:
#     #     continue
#     unsat_cnf_pathes.append(f)

# # random.shuffle(unsat_cnf_pathes)

# cnf_pathes =  sat_cnf_pathes[:300] + unsat_cnf_pathes[:300]
