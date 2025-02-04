import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from scipy.linalg import qr
from qiskit.quantum_info import random_statevector
from ..execute_engine.circuit_converter import qiskit_circuit_to_layer_cirucit

def get_proj_unitary(target_state):
    """Get the projection matrix of the target state."""
    # orthgonal_basis = get_orthgonal_basis(target_state)
    N = target_state.shape[0]
    ## clip the target state
    ## if some elements are too small, we set them to zero
    # target_state[np.abs(target_state) < 1e-10] = 0
    # target_stateconj = target_state.conj()
    unitary = np.zeros((N,N),dtype=np.complex128)
    Q, R = qr(np.column_stack((target_state, np.eye(N))), mode='economic')
    unitary[:, 0] = target_state
    # Ensure that the first column of Q is |phi> (up to a global phase, which doesn't matter)
    if np.allclose(Q[:, 0],target_state, atol=1e-10):
        unitary[:, 1:] = Q[:, 1:]
    elif np.allclose(Q[:, 0], - target_state, atol=1e-10):  # The phase might be flipped
        unitary[:, 1:] = -Q[:, 1:]
    else:
        raise ValueError("Gram-Schmidt process did not produce |phi> as the first vector.")
    assert np.allclose(np.dot(unitary.conj().T, unitary), np.eye(N))
    assert np.allclose(np.dot(unitary.conj().T,target_state), np.eye(N)[:,0])
    assert np.allclose(np.dot(unitary,np.eye(N)[:,0]), target_state)
    return unitary.conj().T

def unitary_to_gates(unitary, output_qubits):
    """Convert the unitary to gates."""
    # Convert the unitary matrix into an Operator
    op = Operator(unitary)

    # Create a quantum circuit
    qc = QuantumCircuit(len(output_qubits))
    qc.unitary(op,range(len(output_qubits)))

    # Decompose the unitary into basic gates
    decomposed_circuit = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)

    # Get the gates
    gates = qiskit_circuit_to_layer_cirucit(decomposed_circuit)
    return gates

def assertion(target_state,output_qubits):
    """Assert that the projection matrix is correct."""
    unitary = get_proj_unitary(target_state)
    decompose_gates = unitary_to_gates(unitary, output_qubits)
    gates = [[{'name':'unitary','qubits':output_qubits,'params':unitary}]]
    ## assert the gates is the right unitary
    # zerostate = np.zeros_like(target_state)
    # zerostate[0] = 1
    # generate_state = ExcuteEngine.excute_on_pennylane(gates[::-1],type='definedinputstatevector',input_state=zerostate,input_qubits=output_qubits)
    # assert np.allclose(generate_state,target_state)
    # state0 = ExcuteEngine.excute_on_pennylane(gates,type='definedinputstatevector',input_state=target_state,input_qubits=output_qubits)

    # assert abs(1-abs(state0[0]))<1e-3
    return gates, decompose_gates 

    
if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    random_unitary = get_proj_unitary(random_statevector(2**10).data)
    end = perf_counter()
    print(f'generate random unitary: {end-start}')
    start = perf_counter()
    gates = unitary_to_gates(random_unitary,list(range(10)))
    end = perf_counter()
    print(f'convert unitary to gates: {end-start}')
