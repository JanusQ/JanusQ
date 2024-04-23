# import logging
# logging.basicConfig(level=logging.WARN)
import ray
ray.init(log_to_driver=False)

from qiskit.quantum_info import random_unitary
from janusq.data_objects.circuit import qiskit_to_circuit
from janusq.data_objects.backend import  LinearBackend
from janusq.analysis.vectorization import RandomwalkModel
from janusq.data_objects.random_circuit import random_circuits
from janusq.analysis.unitary_decompostion import decompose
import time

n_qubits = 5
backend = LinearBackend(n_qubits, 1, basis_two_gates = ['crz'])

# generate a random unitary
unitary = random_unitary(2**n_qubits).data

# apply decomposition
start_time = time.time()
quct_circuit = decompose(unitary, allowed_dist = 0.2, backend = backend,  multi_process = True)
quct_time = time.time() - start_time


print(quct_time)

print(quct_circuit)