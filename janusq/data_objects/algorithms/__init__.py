import random

from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.random import random_circuit

from janusq.data_objects.backend import Backend
from janusq.data_objects.circuit import Circuit, qiskit_to_circuit

from . import hamiltonian_simulation, vqc, ising, qft, qknn, qsvm, swap, vqe, QAOA_maxcut, grover, deutsch_jozsa, multiplier, qec_5_x, qnn, qugan, simon, square_root, ghz


def get_data(id, qiskit_circuit: QuantumCircuit, mirror, backend: Backend, should_transpile=True, unitary = False):
    new_qiskit_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    for instruction in qiskit_circuit:
        if instruction.operation.name in ('id',):
            continue
        new_instruction = instruction.copy()
        new_instruction.qubits = ()
        qubit_list = []
        for qubit in instruction.qubits:
            if qubit.register.name != 'q':
                qubit_list.append(
                    Qubit(register=QuantumRegister(name='q', size=qubit.register.size), index=qubit.index))
            else:
                qubit_list.append(qubit)
        new_instruction.qubits = tuple(qubit_list)
        new_qiskit_circuit.append(new_instruction)

    if should_transpile == True:
        qiskit_circuit = transpile(new_qiskit_circuit, basis_gates=backend.basis_gates, coupling_map=backend.coupling_map,
                                   optimization_level=3)  # , inst_map=list(i for i in range(18))
    else:
        qiskit_circuit = new_qiskit_circuit

    if mirror:
        qiskit_circuit = qiskit_circuit.compose(qiskit_circuit.inverse())
        
    if unitary:
        from qiskit import Aer, execute
        # 使用 Aer 的 unitary_simulator
        simulator = Aer.get_backend('unitary_simulator')

        # 执行量子电路并获取酉矩阵
        result = execute(qiskit_circuit, simulator).result()
        unitary = result.get_unitary(qiskit_circuit)
        return unitary.data
    else:
        circuit = qiskit_to_circuit(qiskit_circuit)
        circuit.name = id
        return circuit


def get_algorithm_circuits(n_qubits, backend: Backend, algs = ['qft', 'hs', 'ising', 'qknn', 'qsvm', 'vqc', 'ghz', 'grover'], unitary = False) -> list[Circuit]:
    circuits = []
    mirror = False

    kwargs = {'mirror': mirror, 'backend': backend, 'should_transpile': True, "unitary": unitary}
    
    if 'qft' in algs:
        circuits.append(get_data(f'qft_{n_qubits}', qft.get_circuit(n_qubits), **kwargs))
    if 'hs' in algs:    
        circuits.append(get_data(f'hs_{n_qubits}', hamiltonian_simulation.get_circuit(n_qubits), **kwargs))
    if 'ising' in algs:    
        circuits.append(get_data(f'ising_{n_qubits}', ising.get_circuit(n_qubits), **kwargs))
    if 'qknn' in algs:
        circuits.append(get_data(f'qknn_{n_qubits}', qknn.get_circuit(n_qubits), **kwargs))
    if 'qsvm' in algs:
        circuits.append(get_data(f'qsvm_{n_qubits}', qsvm.get_circuit(n_qubits), **kwargs))
    if 'vqc' in algs:
        circuits.append(get_data(f'vqc_{n_qubits}', vqc.get_circuit(n_qubits), **kwargs))
    if 'ghz' in algs:
        circuits.append(get_data(f'ghz_{n_qubits}', ghz.get_circuit(n_qubits), **kwargs))
    if 'grover' in algs:
        circuits.append(get_data(f'grover_{n_qubits}', grover.get_circuit(n_qubits), **kwargs))

    return circuits


def ibu_response_matrix(n_qubits, backend: Backend,measuer_bit,i) -> list[Circuit]:

    circuits = []
    mirror = False
    kwargs = {'mirror': mirror, 'backend': backend, 'should_transpile': True}
    if i == 0:
        qc_t_0 = QuantumCircuit(n_qubits)
        qc_t_0.x(measuer_bit) 
        qc_t_0.x(measuer_bit) 
        circuits.append(get_data(f'qc_t', qc_t_0, **kwargs)) 
    else: 
        qc_t_1 = QuantumCircuit(n_qubits)
        qc_t_1.x(measuer_bit) 
        circuits.append(get_data(f'qc_t', qc_t_1, **kwargs)) 


    return circuits