import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


from janusq.hyqsat.pyTest import solveByHyqsat, solveByMinisat
from janusq.hyqsat.common.read_cnf import readCNF
from janusq.hyqsat.common.reduce import reduce_to_3sat