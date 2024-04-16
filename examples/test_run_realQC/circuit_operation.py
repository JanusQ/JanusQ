from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

GATE_NAME_MAP = {
    'h': 'H',
    'rx': 'X',
    'ry': 'Y',
    'rz': 'Z',
    'cz': 'CZ'
}

QUBIT_NAME = ['q1_1', 'q1_3', 'q1_5', 'q3_5', 'q5_5', 'q5_3', 'q7_3', 'q9_3', 'q11_3', 'q11_1']


def my_circuit_to_dag(circuit: QuantumCircuit):
    instructions = []
    dagnodes = []

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    # for instruction, qargs, cargs in circuit.data:
    # for instruction in circuit:
    for instruction in circuit.data:
        operation = instruction.operation

        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        if operation.name == 'barrier':
            continue
        instructions.append(instruction)  # TODO: 这个里面会不会有barrier
        dagnodes.append(dag_node)
        operation._index = len(dagnodes) - 1
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes

def layer2circuit(layer2instructions, n_qubits):
    new_circuit = QuantumCircuit(n_qubits)
    for layer, instructions in enumerate(layer2instructions):
        involved_qubits = []
        for instruction in instructions:
            involved_qubits += [qubit.index for qubit in instruction.qubits]
            new_circuit.append(instruction)
        new_circuit.barrier()
    return new_circuit


def get_layered_instructions(circuit):
    '''
    这个layer可能不是最好的，应为这个还考虑了画图的时候不要交错
    '''
    dagcircuit, instructions, nodes = my_circuit_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  # Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(node, DAGOpNode) and node.op.name != 'barrier']
        if len(layer) != 0:
            layer2operations.append(layer)

    for _index, instruction in enumerate(instructions):
        assert instruction.operation.name != 'barrier'
        assert nodes[_index].op.name != 'barrier'

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):  # 层号，该层操作
        layer_instructions = []
        for node in operations:  # 该层的一个操作
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer  # instruction在第几层
        layer2instructions.append(layer_instructions)

    return layer2instructions, instruction2layer, instructions, dagcircuit, nodes


from dynamic_decoupling import dynamic_decoupling


def parse_circuit(circuit, devide=True, require_decoupling=True, insert_probs=1):
    circuit_info = {}
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(
        circuit)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的

    if devide:
        layer2instructions = divide_layer(layer2instructions)
        assert_devide(layer2instructions)

    if require_decoupling:
        layer2instructions = dynamic_decoupling(layer2instructions, insert_probs=insert_probs, divide=devide)

    if devide or require_decoupling:
        circuit = layer2circuit(layer2instructions, circuit.num_qubits)
        layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(circuit)

    circuit_info['layer2instructions'] = layer2instructions
    circuit_info['instruction2layer'] = instruction2layer
    circuit_info['instructions'] = instructions
    circuit_info['qiskit_circuit'] = circuit
    return circuit_info


def assert_devide(layer2instructions):
    for layer in layer2instructions:
        s = True
        d = True
        for instruction in layer:
            if len(instruction.qubits) == 1:
                s = False
            else:
                d = False

        assert s == True or d == True


def divide_layer(layer2instructions):
    copy_list = []
    for layer in layer2instructions:
        list_s = []
        list_t = []
        for instruction in layer:
            if len(instruction.qubits) == 1:
                list_s.append(instruction)
            else:
                list_t.append(instruction)
        if len(list_s) != 0 and len(list_t) != 0:
            copy_list.append(list_s)
            copy_list.append(list_t)
        else:
            copy_list.append(layer)
    return copy_list
