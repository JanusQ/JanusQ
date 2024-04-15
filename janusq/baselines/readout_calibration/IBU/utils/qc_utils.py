import numpy as np
from qiskit.converters import circuit_to_dag


def get_response_matrix(backend, qubit_num):
    prop = backend.properties().__dict__
    p_0_1 = prop["_qubits"][qubit_num]['prob_meas1_prep0'][0]
    p_1_0 = prop["_qubits"][qubit_num]['prob_meas0_prep1'][0]
    mat = np.array([[1-p_0_1, p_1_0], [p_0_1, 1-p_1_0]])
    return mat

def get_response_matrix_from_dict(prop, qubit_num):
    p_0_1 = prop["_qubits"][qubit_num]['prob_meas1_prep0'][0]
    p_1_0 = prop["_qubits"][qubit_num]['prob_meas0_prep1'][0]
    mat = np.array([[1 - p_0_1, p_1_0], [p_0_1, 1 - p_1_0]])
    return mat


def active_qubits(qc):
    qc.remove_final_measurements()
    return [qubit.index for qubit in qc.qubits
            if qubit not in circuit_to_dag(qc).idle_wires()]


def get_active_qubits_from_ghz_circuit(qc):
        ops = []
        for x in qc.data:
            if x.operation.name == 'cx':
                ops += [x.qubits[0].index, x.qubits[1].index]
        return tuple(set(ops))
