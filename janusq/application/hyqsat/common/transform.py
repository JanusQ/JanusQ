import logging
from read_cnf import readCNF
import os
from reduce import reduce_to_3sat



dir_name = '../../mul_cnf'
for file in os.listdir(dir_name):
    print(file)
    try:
        clauses, vars = readCNF(os.path.join(dir_name, file))
        reduce_to_3sat(clauses, len(vars), len(clauses), file + "_handle.cnf")
    except Exception as e:
        logging.exception(e)
