import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


from janusq.hyqsat.solver import solve_by_janusct, solve_by_minisat
from janusq.hyqsat.common.read_cnf import readCNF
from janusq.hyqsat.common.reduce import reduce_to_3sat