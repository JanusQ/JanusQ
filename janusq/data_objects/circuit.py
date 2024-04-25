'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 06:05:07
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-19 03:06:03
FilePath: /JanusQ/janusq/data_objects/circuit.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
import copy
import logging
import uuid
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from functools import lru_cache, reduce
from copy import deepcopy

class Gate(dict):
    def __init__(self, gate: dict, layer_index: int = None, copy = True):
        assert 'qubits' in gate
        assert 'name' in gate
        if copy:
            gate = deepcopy(gate)
        self.layer_index = layer_index
        self.index: int = None
        self.vec = None
        super().__init__(gate)

    @property
    def qubits(self):
        return self['qubits']

    @property
    def name(self):
        return self['name']

    @property
    def params(self):
        return self['params']


class Layer(list):
    def __init__(self, gates: list[Gate], layer_index: int = None, copy=True):
        if copy:
            gates = [
                Gate(gate, layer_index, copy = copy)
                for gate in gates
            ]
        super().__init__(gates)


class Circuit(list):
    '''TODO: the circuit should be constant'''

    def __init__(self, layers: list[Layer], n_qubits = None, copy=True, measured_qubits = None, operated_qubits = None):  # , n_qubits=None
        # TODO: use id for cache
        self.id = uuid.uuid1()
        if isinstance(layers, Circuit) and n_qubits is None:
            n_qubits = layers.n_qubits
            
        if isinstance(layers, QuantumCircuit):
            if n_qubits is None:
                n_qubits = layers.num_qubits
            layers = qiskit_to_circuit(layers)

        if copy:
            layers = [
                Layer(layer, index, copy)
                for index, layer in enumerate(layers)
            ]

        super().__init__(layers)

        self.gates: list[Gate] = self._sort_gates()

        if n_qubits is None:
            n_qubits = max(reduce(lambda a, b: a + b,
                           [gate['qubits'] for gate in self.gates]))
        self.n_qubits = n_qubits

        if operated_qubits is None:
            self.operated_qubits = self._sort_qubits()  # benchmarked qubits
        else:
            self.operated_qubits = operated_qubits

        self._assign_index()
        
        self.name : str = None
        # if measured_qubits is None:
        #     self.measured_qubits = self.operated_qubits
        # else:
        self.measured_qubits: list[int] = measured_qubits
    

    def rx(self, angle, qubit, layer_index):
        if layer_index >= len(self):
            self.append(Layer([Gate({'name': 'rx', 'qubits': [qubit], 'params': [angle]})]))
        else:
            self[layer_index].append(Gate({'name': 'rx', 'qubits': [qubit], 'params': [angle]}))

    def crz(self, angle, qubit1, qubit2, layer_index):
        if layer_index >= len(self):
            self.append(Layer([Gate({'name': 'crz', 'qubits': [qubit1, qubit2], 'params': [angle]})]))
        else:
            self[layer_index].append(Gate({'name': 'crz', 'qubits': [qubit1, qubit2], 'params': [angle]}))


    def cx(self, qubit1, qubit2, layer_index):
        if layer_index >= len(self):
            self.append(Layer([Gate({'name': 'cx', 'qubits': [qubit1, qubit2], 'params': []})]))
        else:
            self[layer_index].append(Gate({'name': 'cx', 'qubits': [qubit1, qubit2], 'params': []}))
            
    def ry(self, angle, qubit, layer_index):
        if layer_index >= len(self):
            self.append(Layer([Gate({'name': 'ry', 'qubits': [qubit], 'params': [angle]})]))
        else:
            self[layer_index].append(Gate({'name': 'ry', 'qubits': [qubit], 'params': [angle]}))

    def x(self, qubit, layer_index):
        if layer_index >= len(self):
            self.append(Layer([Gate({'name': 'x', 'qubits': [qubit], 'params': []})]))
        else:
            self[layer_index].append(Gate({'name': 'x', 'qubits': [qubit], 'params': []}))

    def h(self, qubit, layer_index):
        if layer_index >= len(self):
            self.append(Layer([Gate({'name': 'h', 'qubits': [qubit], 'params': []})]))
        else:
            self[layer_index].append(Gate({'name': 'h', 'qubits': [qubit], 'params': []}))

    @property
    def num_two_qubit_gate(self):
        cnt = 0
        for gate in self.gates:
            if len(gate.qubits) > 1:
                cnt+=1
        return cnt
    
    @property
    def duration(self, single_qubit_gate_duration= 30, two_qubit_gate_duration=60):
        layer_types = [max([len(gate.qubits) for gate in layer]) for layer in self]
        duration = 0
        for layer_type in layer_types:
            if layer_type == 1:
                duration += single_qubit_gate_duration
            elif layer_type == 2:
                duration += two_qubit_gate_duration
        return duration
    
    @property
    def depth(self):
        return len(self)
    
    @property
    def n_gates(self):
        return len(self.gates)

    def to_qiskit(self, barrier=True) -> QuantumCircuit:
        return circuit_to_qiskit(self, barrier=barrier)

    def __str__(self) -> str:
        return str(self.to_qiskit())

    def __add__(self, other: list[Layer]):
        if isinstance(other, Circuit):
            n_qubits = max([self.n_qubits, other.n_qubits])
        else:
            n_qubits = self.n_qubits
        return Circuit(list.__add__(self.copy(), other.copy()), n_qubits)

    def copy(self):
        return copy.deepcopy(self)



    def _sort_gates(self,):
        self.gates: list[Gate] = [
            gate
            for layer in self
            for gate in layer
        ]
        self._assign_index()
        return self.gates

    def _sort_qubits(self,):
        operated_qubits = []
        for gate in self.gates:
            operated_qubits += gate.qubits
        self.operated_qubits = list(set(operated_qubits))  # benchmarked qubits
        return self.operated_qubits

    def _assign_index(self):
        for index, gate in enumerate(self.gates):
            gate.index = index
            
    def _assign_layer_index(self):
        for layer_index, layer in enumerate(self):
            for gate in layer:
                gate.layer_index = layer_index   
                
    def clean_empty_layer(self):
        while [] in self:
            self.remove([])
        self._assign_layer_index()
            
    def get_available_space(self, target_gate: Gate):
        assert target_gate in self.gates
        gate_qubits = target_gate.qubits

        if target_gate.layer_index != 0:
            former_layer_index = target_gate.layer_index - 1
            while True:
                now_layer = self[former_layer_index]
                layer_qubits = reduce(lambda a, b: a+b, [gate['qubits'] for gate in now_layer])
                if any([qubit in layer_qubits for qubit in gate_qubits]) or former_layer_index == 0:
                    break
                former_layer_index -= 1
        else:
            former_layer_index = target_gate.layer_index

        if target_gate.layer_index != len(self)-1:
            next_layer_index = target_gate.layer_index + 1
            while True:
                now_layer = self[next_layer_index]
                layer_qubits = reduce(lambda a, b: a+b, [gate['qubits'] for gate in now_layer])
                if any([qubit in layer_qubits for qubit in gate_qubits]) or next_layer_index == len(self)-1:
                    break
                next_layer_index += 1
        else:
            next_layer_index = target_gate.layer_index

        return range(former_layer_index, next_layer_index)
    
   
    def move(self, gate: Gate, new_layer: int):
        assert gate in self.gates
  
        new_circuit = self.copy()
        new_gate: Gate = new_circuit.gates[gate.index]
        now_layer: Layer = new_circuit[new_gate.layer_index]
        now_layer.remove(new_gate)
        
        new_circuit[new_layer].append(new_gate)
        new_gate.layer_index = new_layer
        
        new_circuit._sort_gates()
        new_circuit._assign_index()
        new_circuit.clean_empty_layer()
        return new_circuit
        


class SeperatableCircuit(Circuit):
    def __init__(self, seperatable_circuits: list[Circuit], n_qubits):
        max_layer = max([len(c) for c in seperatable_circuits])

        overall_circuit = []
        for layer_index in range(max_layer):
            overall_layer = Layer([])
            for c in seperatable_circuits:
                if len(c) > layer_index:
                    for gate in c[layer_index]:
                        overall_layer.append(gate)

            overall_circuit.append(overall_layer)

        self.seperatable_circuits = seperatable_circuits

        super().__init__(overall_circuit, n_qubits, copy=False)


'''
    Circuit of JanusQ is represented of a 2-d list, named layer_circuit:
    [
        [{
            'name': 'CX',
            'qubits': [0, 1],
            'params': [],
        },{
            'name': 'RX',
            'qubits': [0],
            'params': [np.pi/2],
        }],
        [{
            'name': 'CX',
            'qubits': [2, 3],
            'params': [],
        }]
    ]
'''


def circuit_to_qiskit(circuit: Circuit, barrier=True) -> QuantumCircuit:
    qiskit_circuit = QuantumCircuit(circuit.n_qubits)

    for layer in circuit:
        for gate in layer:
            name = gate['name']
            qubits = gate['qubits']
            params = gate['params']
            if name in ('rx', 'ry', 'rz'):
                assert len(params) == 1 and len(qubits) == 1
                qiskit_circuit.__getattribute__(
                    name)(float(params[0]), qubits[0])
            elif name in ('cz', 'cx'):
                assert len(params) == 0 and len(qubits) == 2
                qiskit_circuit.__getattribute__(name)(qubits[0], qubits[1])
            elif name in ('h', 'x'):
                qiskit_circuit.__getattribute__(name)(qubits[0])
            elif name in ('u', 'u3', 'u1', 'u2'):
                '''TODO: 参数的顺序需要check下， 现在是按照pennylane的Rot的'''
                qiskit_circuit.__getattribute__(name)(
                    *[float(param) for param in params], qubits[0])
            elif name in ('crz',):
                qiskit_circuit.__getattribute__(name)(
                    *[float(param) for param in params], *qubits)
            else:
                # circuit.__getattribute__(name)(*(params + qubits))
                logging.error('unkown gate', gate)
                # raise Exception('unkown gate', gate)

        if barrier:
            qiskit_circuit.barrier()

    return qiskit_circuit


def qiskit_to_circuit(qiskit_circuit: QuantumCircuit) -> Circuit:
    '''
    description: convert a qiskiut circuit to our format circuit 
    param {QuantumCircuit} qiskit_circuit:
    return {Circuit}
    '''
    layer_to_qiskit_instructions = _get_layered_instructions(qiskit_circuit)[0]

    layer_to_instructions = []
    for layer_instructions in layer_to_qiskit_instructions:
        layer_instructions = [_instruction_to_gate(
            instruction) for instruction in layer_instructions]
        layer_to_instructions.append(layer_instructions)

    return Circuit(layer_to_instructions, n_qubits=qiskit_circuit.num_qubits)


def _instruction_to_gate(instruction: Instruction):
    name = instruction.operation.name
    parms = list(instruction.operation.params)
    return {
        'name': name,
        'qubits': [qubit.index for qubit in instruction.qubits],
        'params': [ele if isinstance(ele, float) else float(ele) for ele in parms],
    }


def _get_layered_instructions(circuit: QuantumCircuit):
    dagcircuit, instructions, nodes = _circuit_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  # Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(
            node, DAGOpNode) and node.op.name not in ('barrier', 'measure')]
        if len(layer) != 0:
            layer2operations.append(layer)

    # for _index, instruction in enumerate(instructions):
    #     assert instruction.operation.name != 'barrier'
    #     assert nodes[_index].op.name != 'barrier'

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):
        layer_instructions = []
        for node in operations:
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer  # layer of instruction
        layer2instructions.append(layer_instructions)

    return layer2instructions, instruction2layer, instructions, dagcircuit, nodes


def _circuit_to_dag(circuit: QuantumCircuit):
    instructions = []
    dagnodes = []

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name if 'name' in circuit else None
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction in circuit.data:
        operation = instruction.operation

        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        if operation.name == 'barrier':
            continue
        instructions.append(instruction)
        dagnodes.append(dag_node)
        # operation._index = len(dagnodes) - 1
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes


def assign_barrier(qiskit_circuit):
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = _get_layered_instructions(
        qiskit_circuit) 

    new_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    for layer, instructions in enumerate(layer2instructions):
        for instruction in instructions:
            new_circuit.append(instruction)
        new_circuit.barrier()

    return new_circuit
