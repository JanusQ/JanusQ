'''
    TODO: 还没有加上quct的部分
'''

import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from data_objects.circuit import qiskit_to_circuit
from data_objects.algorithms import get_algs, qft, ghz
from data_objects.backend import FullyConnectedBackend, LinearBackend
import math
import numpy as np
from scipy.stats import unitary_group
import time
from qiskit import transpile
import cloudpickle as pickle
from qiskit.quantum_info import Operator
import logging
from tools.saver import dump
from analysis.unitary_decompostion import circuit_to_matrix, decompose


np.random.RandomState()

# TODO: 现在还不支持任意的backend
allowed_dist = 0.1
n_qubits = 5

alg_backend = FullyConnectedBackend(n_qubits=n_qubits, basis_single_gates=[
                                'u'], basis_two_gates=['cz'])

circuit = get_algs(n_qubits, alg_backend, algs = ['qft'])[0]

backend = FullyConnectedBackend(n_qubits=n_qubits, basis_single_gates=[
                                'u'], basis_two_gates=['crz'])
target_U = circuit_to_matrix(circuit, n_qubits)

decomposed_cirucit = decompose(target_U, backend=backend, allowed_dist=allowed_dist,
                     multi_process=True, logger_level=logging.DEBUG)
# decomposed_cirucit = transpile(decomposed_cirucit.to_qiskit(), basis_gates=alg_backend.basis_gates, optimization_level=3)

# print(circuit.n_gates, len(decomposed_cirucit))
logging.info(circuit.n_gates, decomposed_cirucit.n_gates)