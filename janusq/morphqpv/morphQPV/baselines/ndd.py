import numpy as np
from scipy.linalg import qr
from copy import deepcopy
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, transpile
from ..execute_engine.circuit_converter import qiskit_circuit_to_layer_cirucit
def is_ndd_type(real_output_state):
    pass

def is_classical_state(state):
    # judge the quantum statevector is a classical state
    return np.sum(np.abs(state)) == 1

def get_unitary(target_state):
    """Get the projection matrix of the target state."""
    # orthgonal_basis = get_orthgonal_basis(target_state)
    N = target_state.shape[0]
    Q, R = qr(np.column_stack((target_state, np.eye(N))), mode='economic')
    Qprime = deepcopy(Q)
    Qprime[:,0] = Q[:,0]
    Qprime[:,1:] = - Q[:,1:]
    unitary= Q @ Qprime.conj().T

    assert np.allclose(np.dot(unitary.conj().T, unitary), np.eye(N))
    assert np.allclose(np.dot(unitary.conj().T,target_state), target_state)
    return unitary.conj().T
def controlled_unitary_to_gates(unitary, output_qubits,control_qubits):
    """Convert the unitary to gates."""
    # Convert the unitary matrix into an Operator
    op = Operator(unitary)

    # Create a quantum circuit
    qc = QuantumCircuit(len(output_qubits))
    qc.unitary(op,range(len(output_qubits)))
    cqcgate= qc.to_gate().control(1)
    qc2 = QuantumCircuit(control_qubits+1)
    qc2.append(cqcgate, output_qubits+[control_qubits])
    # Decompose the unitary into basic gates
    decomposed_circuit = transpile(qc2, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
    # Get the gates
    gates = qiskit_circuit_to_layer_cirucit(decomposed_circuit)
    return gates

def assertion(target_state,output_qubits):
    unitary = get_unitary(target_state)
    ancliqubit = max(output_qubits)+1
    decompose_gates = controlled_unitary_to_gates(unitary, output_qubits,ancliqubit)
    decompose_gates = [[{'name':'h','qubits':[ancliqubit]}]]+ decompose_gates +[[{'name':'h','qubits':[ancliqubit]}]]
    
    gates = [[{'name':'h','qubits':[ancliqubit]}],
        [{'name':'ctrl','qubits':output_qubits+[ancliqubit],'params':unitary,'ctrled_qubits':output_qubits,'ctrl_qubits':ancliqubit}],
         [{'name':'h','qubits':[ancliqubit]}]
         ]

    return gates,decompose_gates,ancliqubit
    


