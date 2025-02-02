import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

from janusq.application.hyqsat.solver.solver import solve_by_hyqsat, solve_by_minisat
from janusq.application.hyqsat.common.read_cnf import readCNF
from janusq.application.hyqsat.common.reduce import reduce_to_3sat