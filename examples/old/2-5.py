
import logging
logging.basicConfig(level=logging.WARN)
import sys
sys.path.append('/home/JanusQ-main/')

import ray
ray.init(log_to_driver=False)
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from janusq.data_objects.circuit import qiskit_to_circuit
from janusq.data_objects.backend import  LinearBackend
from janusq.analysis.vectorization import RandomwalkModel
from janusq.data_objects.random_circuit import random_circuits
from janusq.analysis.unitary_decompostion import decompose, U2VModel
import time

n_qubits = 5
backend = LinearBackend(n_qubits, 1, basis_two_gates = ['crz'])

n_step = 2

dataset = random_circuits(backend, n_circuits=50, n_gate_list=[30, 50, 100], two_qubit_prob_list=[.4], reverse=True)


vec_model = RandomwalkModel(
    n_step, 4 ** n_step, backend, directions=('parallel', 'next'))
vec_model.train(dataset, multi_process=True,
                        remove_redundancy=False)

u2v_model = U2VModel(vec_model)
data = u2v_model.construct_data(dataset, multi_process=False)
u2v_model.train(data, n_qubits)


unitary = random_unitary(2**n_qubits).data

# apply decomposition
start_time = time.time()
quct_circuit = decompose(unitary, allowed_dist = 0.2, backend = backend, u2v_model = u2v_model, multi_process = True)
quct_time = time.time() - start_time

quct_circuit,quct_time


# compare it with the qsd method
from qiskit.synthesis.unitary.qsd import qs_decomposition

start_time =time.time()
qc = qs_decomposition(unitary)

qsd_circuit = qiskit_to_circuit(qc)
qsd_time = time.time() - start_time

synthesis_method_result = [qsd_circuit,  quct_circuit]
synthesis_method_time = [qsd_time,  quct_time]
for res, tim in zip(synthesis_method_result, synthesis_method_time):
    # print(res, tim)
    print(f"#gate: {res.n_gates}, #two_qubit_gate: {res.num_two_qubit_gate}, depth: {res.depth}, time: {tim} \n")


