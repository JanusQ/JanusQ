from numpy import pi
from qiskit import QuantumCircuit


def qft_rotations(circuit: QuantumCircuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(pi / 2 ** (n - qubit), qubit, n)
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
        qft_rotations(circuit, n)

def swap_registers(circuit: QuantumCircuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit

def get_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qft_rotations(qc, n_qubits)
    swap_registers(qc, n_qubits)
    return qc