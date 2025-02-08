import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

from .solver.solver import solve_by_hyqsat, solve_by_minisat
from .common.read_cnf import readCNF
from .common.reduce import reduce_to_3sat