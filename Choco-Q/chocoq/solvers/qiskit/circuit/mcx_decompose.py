from qiskit import QuantumCircuit

# 没用上
def mcx_n_anc_linear_decompose(circuit: QuantumCircuit, control_qubits, target_qubit, ancillary_qubits):
    """
    This function implements the multi-controlled-X gate using the Toffoli gate.
    """
    if len(control_qubits) == 0:
        circuit.x(target_qubit)
    elif len(control_qubits) == 1:
        circuit.cx(control_qubits[0], target_qubit)
    elif len(control_qubits) == 2:
        circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
    else:
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        for i in range(len(control_qubits) - 2):
            circuit.ccx(ancillary_qubits[i], control_qubits[i + 2], ancillary_qubits[i + 1])
        circuit.ccx(ancillary_qubits[len(control_qubits) - 2], control_qubits[-1], target_qubit)
        # reverse the process
        for i in list(range(len(control_qubits) - 2))[::-1]:
            circuit.ccx(ancillary_qubits[i], control_qubits[i + 2], ancillary_qubits[i + 1])
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])

# 用更多比特，对应cxmode为linear，拓扑可能更差
def mcx_n_anc_log_decompose(circuit: QuantumCircuit, control_qubits, target_qubit, ancillary_qubits):
    """
    This function implements the multi-controlled-X gate using the Toffoli gate.
    """
    if len(control_qubits) == 0:
        circuit.x(target_qubit)
    elif len(control_qubits) == 1:
        circuit.cx(control_qubits[0], target_qubit)
        return 0
    elif len(control_qubits) == 2:
        circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        return 1
    else:
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        res = mcx_n_anc_log_decompose(
            circuit,
            control_qubits[2:] + [ancillary_qubits[0]],
            target_qubit,
            ancillary_qubits[1:],
        )
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        return res