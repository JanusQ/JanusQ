'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-16 13:33:36
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-19 01:55:00
FilePath: /JanusQ/janusq/data_objects/algorithms/__init__.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''

from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Qubit
from janusq.objects.backend import Backend
from janusq.objects.circuit import Circuit, qiskit_to_circuit
from . import hamiltonian_simulation, vqc, ising, qft, qknn, qsvm, grover,  ghz


def get_data(id, qiskit_circuit: QuantumCircuit, mirror, backend: Backend, should_transpile=True, unitary = False):
    '''
    description: 
    param {str} id: algorithm name 
    param {QuantumCircuit} qiskit_circuit: algorithm circuit 
    param {bool} mirror: Whether to add the reverse circuit of the circuit to the original circuit
    param {Backend} backend: algorithm transpile to fit backend
    param {bool} should_transpile: weather to transpile the algorithm circuit
    param {bool} unitary: weather to return algorithm unitary
    '''
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
        simulator = Aer.get_backend('unitary_simulator')

        # Execute quantum circuits and obtain unitary matrices
        result = execute(qiskit_circuit, simulator).result()
        unitary = result.get_unitary(qiskit_circuit)
        return unitary.data
    else:
        circuit = qiskit_to_circuit(qiskit_circuit)
        circuit.name = id
        return circuit


def get_algorithm_circuits(n_qubits: int, backend: Backend, algs = ['qft', 'hs', 'ising', 'qknn', 'qsvm', 'vqc', 'ghz', 'grover'], unitary:bool = False) -> list[Circuit]:
    '''
    description: get specified algorithm circuit
    param {int} n_qubits: the number of qubit
    param {Backend} backend: algorithm transpile to fit backend
    param {list[str]} algs: specifed algorithm
    param {bool} unitary: weather to return algorithm unitary
    return {list[Circuit]} circuits: all algorithm circuits
    '''
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
    """
    Generate circuits for IBU response matrix.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        backend (Backend): The backend used for simulation.
        measuer_bit (int): The qubit index for measurement.
        i (int): Index parameter.

    Returns:
        list[Circuit]: List of circuits for IBU response matrix.
    """

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