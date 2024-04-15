import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from data_objects.random_circuit import random_circuits
from data_objects.backend import Backend, LinearBackend, GridBackend, FullyConnectedBackend

n_qubits = 5

backend = LinearBackend(n_qubits, 1)
circuits = random_circuits(backend, 100, [30, 50, 100], [.4, .5], True)

# for circuit in circuits:
#     print(circuit.qiskit_circuit())

n_columns, n_rows = 3, 4
backend = GridBackend(n_columns, n_rows, 1)
circuits = random_circuits(backend, 100, [30, 50, 100], [.4, .5], True)


backend = FullyConnectedBackend(n_qubits)
circuits = random_circuits(backend, 100, [30, 50, 100], [.4, .5], True)