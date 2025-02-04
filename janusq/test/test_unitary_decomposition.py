import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from analysis.unitary_decompostion import decompose
from tools.saver import dump


import logging
from qiskit.quantum_info import Operator
import cloudpickle as pickle
from qiskit import transpile
import time
from scipy.stats import unitary_group
import numpy as np
import math
from objects.backend import FullyConnectedBackend, LinearBackend


allowed_dist = 0.1
n_qubits = 5

np.random.RandomState()
target_U = unitary_group.rvs(2**n_qubits)

n_qubits = int(math.log2(len(target_U)))


backend = FullyConnectedBackend(n_qubits=n_qubits, basis_single_gates=['u'], basis_two_gates=['crz'])


start_time = time.time()

solution = decompose(target_U, backend=backend, allowed_dist=allowed_dist, multi_process=True, logger_level=logging.DEBUG)

decomposed_result = {
    'n_qubits': n_qubits,
    'target_U': target_U,
    'solution': solution,
    'synthesis_time': time.time() - start_time,
    'allowed_dist': allowed_dist,
}

with open(f'./temp_daya/decomposed_{n_qubits}_random_{str(np.random.randint(0, 10000))}.pkl', 'wb') as file:
    dump(decomposed_result, file)


