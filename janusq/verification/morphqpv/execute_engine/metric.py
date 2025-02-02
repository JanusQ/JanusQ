
import numpy as np
def _funm_svd(matrix, func):
    """Apply real scalar function to singular values of a matrix.

    Args:
        matrix (array_like): (N, N) Matrix at which to evaluate the function.
        func (callable): Callable object that evaluates a scalar function f.

    Returns:
        ndarray: funm (N, N) Value of the matrix function specified by func
                 evaluated at `A`.
    """
    import scipy.linalg as la

    unitary1, singular_values, unitary2 = la.svd(matrix)
    diag_func_singular = np.diag(func(singular_values))
    return unitary1.dot(diag_func_singular).dot(unitary2)

def fidelity(sigma,rho):
    s1sq = _funm_svd(sigma, np.sqrt)
    s2sq = _funm_svd(rho, np.sqrt)
    fid = np.linalg.norm(s1sq.dot(s2sq), ord="nuc") ** 2
    return float(np.real(fid))


from .circuit_converter import *
from qiskit import transpile
from qiskit.providers.fake_provider import FakeTokyo
def latency(layer_circuit,N_qubit):
    qiskit_circuit = layer_circuit_to_qiskit_circuit(layer_circuit,N_qubit)
    backend = FakeTokyo()
    qiskit_circuit = transpile(qiskit_circuit, backend=backend)
    latency_qubits = {q:0 for q in qiskit_circuit.qubits}
    for gate in qiskit_circuit:
        gate_lenth = backend.properties().gate_length(gate.operation.name,[q.index for q in gate.qubits] )
        for q in gate.qubits:
            latency_qubits[q] += gate_lenth
    return max(latency_qubits.values())