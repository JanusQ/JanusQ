import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile
from scipy.linalg import expm
from typing import List, Union, Tuple, Iterable

from .mcx_decompose import mcx_n_anc_log_decompose


def apply_convert(qc: QuantumCircuit, list_qubits, bit_string):
    num_qubits = len(bit_string)
    for i in range(0, num_qubits - 1):
        qc.cx(list_qubits[i + 1], list_qubits[i])
        if bit_string[i] == bit_string[i + 1]:
            qc.x(list_qubits[i])
    qc.h(list_qubits[num_qubits - 1])
    qc.x(list_qubits[num_qubits - 1])

def apply_reverse(qc: QuantumCircuit, list_qubits, bit_string):
    num_qubits = len(bit_string)
    qc.x(list_qubits[num_qubits - 1])
    qc.h(list_qubits[num_qubits - 1])
    for i in range(num_qubits - 2, -1, -1):
        if bit_string[i] == bit_string[i + 1]:
            qc.x(list_qubits[i])
        qc.cx(list_qubits[i + 1], list_qubits[i])

def mcx_gate_decompose(qc: QuantumCircuit, list_controls:Iterable, qubit_target:int, list_ancilla:Iterable, mcx_mode):
    if mcx_mode == 'constant':
        # 自动分解，34 * 非零元
        qc.mcx(list_controls, qubit_target, list_ancilla[0], mode='recursion')
    elif mcx_mode == 'linear':
        # log 但是用更多比特，拓扑反而更差
        mcx_n_anc_log_decompose(qc,list_controls, qubit_target, list_ancilla)
    else:
        qc.mcx(list_controls, qubit_target, list_ancilla[0])

def decompose_phase_gate(qc: QuantumCircuit, list_qubits:list, list_ancilla:list, phase:float, mcx_mode) -> QuantumCircuit:
    """
    Decompose a phase gate into a series of controlled-phase gates.
    Args:
        qc
        list_qubits
        list_ancilla
        phase (float): the phase angle of the phase gate.
        mcx_mode (str): the type of ancillary qubits used in the controlled-phase gates.
            'constant': use a constant number of ancillary qubits for all controlled-phase gates.
            'linear': use a linear number of ancillary qubits to guarantee logarithmic depth.
    Returns:
        QuantumCircuit: the circuit that implements the decomposed phase gate.
    """
    num_qubits = len(list_qubits)
    if num_qubits == 1:
        qc.p(phase, list_qubits[0])
    elif num_qubits == 2:
        qc.cp(phase, list_qubits[0], list_qubits[1])
    else:
        # convert into the multi-cx gate 
        # partition qubits into two sets
        half_num_qubit = num_qubits // 2
        qr1 = list_qubits[:half_num_qubit]
        qr2 = list_qubits[half_num_qubit:]
        qc.rz(-phase/2, list_ancilla[0])
        # use ", mode='recursion'" without transpile will raise error 'unknown instruction: mcx_recursive'
        mcx_gate_decompose(qc, qr1, list_ancilla[0], list_ancilla[1:], mcx_mode)
        qc.rz(phase/2, list_ancilla[0])
        mcx_gate_decompose(qc, qr2, list_ancilla[0], list_ancilla[1:], mcx_mode)
        qc.rz(-phase/2, list_ancilla[0])
        mcx_gate_decompose(qc, qr1, list_ancilla[0], list_ancilla[1:], mcx_mode)
        qc.rz(phase/2, list_ancilla[0])
        mcx_gate_decompose(qc, qr2, list_ancilla[0], list_ancilla[1:], mcx_mode)

# separate functions facilitate library calls
def driver_component(qc: QuantumCircuit, list_qubits:Iterable, list_ancilla:Iterable, bit_string:str, phase:float, mcx_mode:str='linear'):
    apply_convert(qc, list_qubits, bit_string)
    decompose_phase_gate(qc, list_qubits, list_ancilla, -phase, mcx_mode)
    qc.x(list_qubits[-1])
    decompose_phase_gate(qc, list_qubits, list_ancilla, phase, mcx_mode)
    qc.x(list_qubits[-1])
    apply_reverse(qc, list_qubits, bit_string)

# -------

def get_driver_component(num_qubits, t, bit_string, use_decompose=True):
    list_qubits = list(range(num_qubits))
    list_ancilla = [num_qubits, num_qubits + 1]
    qc = QuantumCircuit(num_qubits)
    apply_convert(qc, list_qubits, bit_string)
    qc.barrier(label='convert')
    if use_decompose == True:
        ancilla = QuantumRegister(2, name='anc')
        qc.add_register(ancilla)
        decompose_phase_gate(qc, list_qubits, list_ancilla, -2 * np.pi * t, 'constant')
    else:
        qc.mcp(-2 * np.pi * t, list_qubits[1:], 0)
    qc.barrier(label='multi-ctrl')
    qc.x(num_qubits - 1)
    if use_decompose == True:
        decompose_phase_gate(qc, list_qubits, list_ancilla, 2 * np.pi * t, 'constant')
    else:
        qc.mcp(2 * np.pi * t, list_qubits[1:], 0)
    qc.x(num_qubits - 1)
    qc.barrier(label='reverse')
    apply_reverse(qc, list_qubits, bit_string)
    return qc

# 输入qc, 返回电路酉矩阵
def get_circ_unitary(quantum_circuit):
    backend = Aer.get_backend('unitary_simulator' )
    new_circuit = transpile(quantum_circuit, backend)
    job = backend.run(new_circuit)
    result = job.result()
    unitary = result.get_unitary()
    from chocoq.utils.linear_system import reorder_tensor_product
    reoder_unitary = reorder_tensor_product(np.array(unitary))
    # 张量积逆序
    return reoder_unitary

def tensor_product(matrices):
    sigma_plus = np.array([[0, 1], [0, 0]])
    sigma_minus = np.array([[0, 0], [1, 0]])
    identity = np.array([[1, 0], [0, 1]])
    mlist = [sigma_plus, sigma_minus, identity]
    result = np.array(mlist[matrices[0]])
    for matrix in matrices[1:]:
        result = np.kron(result, mlist[matrix])
    return result

def get_simulate_unitary(t, bit_string):    
    # for unitary build contrary_v
    U = expm(-1j * 2 * np.pi * t * (tensor_product(bit_string)+tensor_product([not(i) for i in bit_string])))
    return U

def decompose_unitary(t, bit_string):
    unitary = get_simulate_unitary(t,bit_string)
    qc = QuantumCircuit(len(bit_string))
    qc.unitary(unitary, range(len(bit_string)))
    return qc.decompose()