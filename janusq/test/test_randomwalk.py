import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from analysis.vectorization import RandomwalkModel

from objects.random_circuit import random_circuits, random_circuit
from objects.backend import Backend, LinearBackend, GridBackend, FullyConnectedBackend

# 把 circuit 和 vecs, paths 分离

n_qubits = 5

backend = LinearBackend(n_qubits, 1)
circuits = random_circuits(backend, 50, [30, 50, 100], [.4, .5], True)

random_walk_model = RandomwalkModel(n_steps = 2, n_walks = 20, backend = backend)
vecs = random_walk_model.train(circuits, multi_process=False, remove_redundancy = False)


circuit = random_circuit(backend, 100, .5, True)
gate_vecs = random_walk_model.vectorize(circuit)

pass