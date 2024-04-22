'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 06:05:11
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-18 07:27:47
FilePath: /JanusQ/janusq/data_objects/random_circuit.py
Description: randomly generate various circuit for different data-driven analysis tasks

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''

import logging
import random
from cmath import pi

from qiskit import QuantumCircuit

from janusq.data_objects.backend import Backend
from janusq.data_objects.circuit import Circuit, SeperatableCircuit, qiskit_to_circuit
import math




def random_circuits(backend: Backend, n_circuits: int, n_gate_list: list, two_qubit_prob_list: list, reverse: bool = True) -> list[Circuit]:
    '''
    description: 
    param {Backend} backend: circuit are generated based on backend
    param {int} n_circuits: total number of circuit to generate 
    param {list} n_gate_list: devide all circuit into several part, each part has ciruits with same number of gates specified in n_gate_list
    param {list} two_qubit_prob_list: ciruits with same number of gates are alse devide into several part which has the same proportion of two qubit gate
    param {bool} reverse: Whether to add the reverse circuit of the circuit to the end of the original circuit
    return {list[Circuit]}: generate circuits
    '''
    circuits = []
    for n_gates in n_gate_list:
        for two_qubit_prob in two_qubit_prob_list:
            for _ in range(math.ceil(n_circuits/len(n_gate_list)/len(two_qubit_prob_list))):
                circuits.append(random_circuit(backend, n_gates, two_qubit_prob, reverse))
    return circuits
    

def random_grouping_scheme(backend: Backend, n_qubits_per_group: int): 

    grouping_scheme = []
    now_group = []
    
    remaining_qubits = list(range(backend.n_qubits))
    while True:
        if len(now_group) == 0:
            now_qubit = random.choice(remaining_qubits)
        else:
            neighbor_qubits = backend.topology[now_qubit]
            neighbor_qubits = [qubit for qubit in neighbor_qubits if qubit in remaining_qubits]
            
            if len(neighbor_qubits) == 0:
                grouping_scheme.append(now_group)
                now_group = []
                continue
            
            now_qubit = random.choice(neighbor_qubits)
        
        now_group.append(now_qubit)
        remaining_qubits.remove(now_qubit)
        
        if len(now_group) == n_qubits_per_group:
            grouping_scheme.append(now_group)
            now_group = []
            
        if len(remaining_qubits) == 0:
            grouping_scheme.append(now_group)
            now_group = []
            break

    return grouping_scheme

def random_seperatable_circuits(backend: Backend, n_circuits: int, n_qubits_per_group: int, n_gate_list: list, two_qubit_prob_list: list, reverse: bool = True) -> list[SeperatableCircuit]:

    seperatable_circuits: list[SeperatableCircuit] = []
    
    for n_gates in n_gate_list:
        for two_qubit_prob in two_qubit_prob_list:
            for _ in range(math.ceil(n_circuits/len(n_gate_list)/len(two_qubit_prob_list))):
                grouping_scheme = random_grouping_scheme(backend, n_qubits_per_group)
                seperatable_circuit = []
                for group in grouping_scheme:
                    sub_backend = backend.get_sub_backend(group)
                    circuit = random_circuit(sub_backend, n_gates//backend.n_qubits*len(group), two_qubit_prob, reverse,)
                    if len(circuit) != 0:
                        seperatable_circuit.append(circuit)
                seperatable_circuits.append(SeperatableCircuit(seperatable_circuit, backend.n_qubits))
  
    return seperatable_circuits



def random_circuit(backend: Backend, n_gates:int, two_qubit_prob:int, reverse:bool = False) -> Circuit:
    '''
    description: generate one citcuit 
    param {Backend} backend: circuit is generated based on backend
    param {int} n_gates: the number of gate
    param {int} two_qubit_prob: the two qubit proportion of circuit
    param {bool} reverse: Whether to add the reverse circuit of the circuit to the end of the original circuit
    return {Circuit}
    '''
    coupling_map, n_qubits = backend.coupling_map, backend.n_qubits
    basis_single_gates, basis_two_gates = backend.basis_single_gates, backend.basis_two_gates
    
    if reverse:
        n_gates = n_gates//2
        
    circuit = QuantumCircuit(n_qubits)
    qubits = backend.involvod_qubits

    circuit = circuit.compose(random_1q_layer(n_qubits, qubits, basis_single_gates))
    
    n_gates -= len(qubits)

    cnt = 0
    while cnt < n_gates:
        random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates)
        cnt += 1

    if reverse:
        circuit = circuit.compose(circuit.inverse())

    return qiskit_to_circuit(circuit)


def random_pi(): 
    rand = round(random.random(), 1)
    if rand == 0: return 0.1 * pi
    return rand * 2 *  pi

def random_gate(circuit: QuantumCircuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates):
    if random.random() < two_qubit_prob:
        gate_type = basis_two_gates[0]
        assert len(basis_two_gates) == 1
    else:
        gate_type = random.choice(basis_single_gates)
    
    if len(coupling_map) != 0:
        operated_qubits = list(random.choice(coupling_map))
        control_qubit = operated_qubits[0]
        target_qubit = operated_qubits[1]
        
    if len(coupling_map) == 0 and gate_type in ('cx', 'cz', 'unitary'):
        gate_type = random.choice(basis_single_gates)
        logging.warning('WARNING: no coupling map')
    
    if gate_type == 'cz':
        # 没有控制和非控制的区别
        circuit.cz(control_qubit, target_qubit)   
    elif gate_type == 'cx':
        circuit.cx(control_qubit, target_qubit) 
    elif gate_type in ('h',):
        selected_qubit = random.choice(qubits)
        circuit.h(selected_qubit)
    elif gate_type in ('crz',):
        selected_qubit = random.choice(qubits)
        circuit.crz(random_pi(), control_qubit, target_qubit)
    elif gate_type in ('rx', 'rz', 'ry'):
        selected_qubit = random.choice(qubits)
        getattr(circuit, gate_type)(random_pi(), selected_qubit)
        # getattr(circuit, gate_type)(pi, selected_qubit)
    elif gate_type in ('u',):
        selected_qubit = random.choice(qubits)
        getattr(circuit, gate_type)(random_pi(), random_pi(), random_pi(), selected_qubit)
    else:
        logging.exception('Unknown gate type', gate_type)
        # raise Exception('Unknown gate type', gate_type)


def random_1q_layer(n_qubits, qubits, basis_single_gates) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    
    for qubit in qubits:
        gate_type = random.choice(basis_single_gates)
        if gate_type in ('h',):
            circuit.h(qubits)
        elif gate_type in ('rx', 'rz', 'ry'):
            getattr(circuit, gate_type)(random_pi(), qubit)
        elif gate_type in ('u',):
            getattr(circuit, gate_type)(random_pi(), random_pi(), random_pi(), qubit)
        else:
            logging.exception('Unknown gate type', gate_type)
            # raise Exception('Unknown gate type', gate_type)
        
    return circuit



def random_cyclic_circuit(backend: Backend, n_gates, two_qubit_prob = 0.5, block_size = 20) -> QuantumCircuit:
    coupling_map, n_qubits = backend.coupling_map, backend.n_qubits
    basis_single_gates, basis_two_gates = backend.basis_single_gates, backend.basis_two_gates
    
    coupling_map = backend.coupling_map
    basis_single_gates = backend.basis_single_gates
    basis_two_gates = backend.basis_two_gates

    block = _random_block(n_qubits, block_size, two_qubit_prob,  coupling_map, basis_single_gates, basis_two_gates,)    
    
    block_num = n_gates // block_size
        
    circuit = QuantumCircuit(n_qubits)

    layer_1q = random_1q_layer(n_qubits, basis_single_gates)
    circuit = circuit.compose(layer_1q)
    circuit.barrier()
    
    for i_block in range(block_num):    
        circuit = circuit.compose(block)
        circuit.barrier()
    
    circuit = circuit.compose(layer_1q.inverse())
       
    return qiskit_to_circuit(circuit)


def _random_block(n_qubits, n_gates, two_qubit_prob,  coupling_map, basis_single_gates, basis_two_gates,):
    circuit = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))
    
    n_gates = n_gates//2


    cnt = 0
    while cnt < n_gates:
        random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates)
        cnt += 1
    circuit = circuit.compose(circuit.inverse())
    
    return circuit


