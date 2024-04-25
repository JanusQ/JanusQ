import ray
ray.init(log_to_driver=False)

from qiskit.quantum_info import random_unitary
from janusq.data_objects.circuit import qiskit_to_circuit
from janusq.data_objects.backend import  LinearBackend
from janusq.analysis.vectorization import RandomwalkModel
from janusq.data_objects.random_circuit import random_circuits
from janusq.analysis.unitary_decompostion import decompose
import time
from janusq.analysis.vectorization import RandomwalkModel, extract_device
from janusq.data_objects.backend import GridBackend, FullyConnectedBackend, LinearBackend

from janusq.dataset import real_qc_5bit
from janusq.data_objects.random_circuit import random_circuits, random_circuit

circuits, fidelities = real_qc_5bit
print(len(circuits))

backend = FullyConnectedBackend(5)
up_model = RandomwalkModel(n_steps = 1, n_walks = 10, backend = backend, circuits = circuits)

from janusq.analysis.fidelity_prediction import FidelityModel
fidelity_model = FidelityModel(up_model)
fidelity_model.train((circuits, fidelities), multi_process = False, max_epoch = 20)
circuit = random_circuit(backend, n_gates = 100, two_qubit_prob = 0.5)
fidelity_model.predict_circuit_fidelity(circuit)


